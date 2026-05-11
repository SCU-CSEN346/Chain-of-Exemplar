import json, os, torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/.hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/.hf_cache"
os.environ["HF_HOME"] = "/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/.hf_cache"

DATA_ROOT = "data/scienceqa"
INP = f"{DATA_ROOT}/problems.json"
OUT = f"{DATA_ROOT}/problems_blip2xl_angle.json"
K = 5
MODEL_NAME = "all-MiniLM-L6-v2"

problems = json.load(open(INP))
train_ids = [k for k, v in problems.items() if v["split"] == "train"]
all_ids = list(problems.keys())

def ans(p):
    return p["choices"][p["answer"]]

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    return " ".join(s.lower().strip().split())

def question_signature(s):
    s = normalize_text(s)

    prefixes = [
        "using only these supplies, which question can ",
    ]

    for prefix in prefixes:
        if s.startswith(prefix) and " investigate with an experiment?" in s:
            return prefix + "<name> investigate with an experiment?"

    return s

def context_text(p):
    parts = []

    hint = p.get("hint", "")
    lecture = p.get("lecture", "")
    solution = p.get("solution", "")

    if isinstance(hint, str) and hint.strip():
        parts.append(hint.strip())
    if isinstance(lecture, str) and lecture.strip():
        parts.append(lecture.strip())
    if isinstance(solution, str) and solution.strip():
        parts.append(solution.strip())

    return " ".join(parts)

def get_modality(p):
    has_image = isinstance(p.get("image"), str) and len(p.get("image").strip()) > 0
    has_text = len(context_text(p).strip()) > 0

    if has_image and has_text:
        return "IMG+TXT"
    elif has_image:
        return "IMG"
    elif has_text:
        return "TXT"
    else:
        return "NO"

model = SentenceTransformer(MODEL_NAME)

train_modalities = [get_modality(problems[i]) for i in train_ids]
all_modalities = [get_modality(problems[i]) for i in all_ids]

print("Encoding train questions...")
train_q = model.encode(
    [problems[i].get("question", "") for i in train_ids],
    batch_size=128,
    convert_to_tensor=True,
    show_progress_bar=True
)

print("Encoding all questions...")
all_q = model.encode(
    [problems[i].get("question", "") for i in all_ids],
    batch_size=128,
    convert_to_tensor=True,
    show_progress_bar=True
)

print("Encoding train answers/contexts...")
train_ans = model.encode(
    [ans(problems[i]) for i in train_ids],
    batch_size=128,
    convert_to_tensor=True,
    show_progress_bar=True
)
train_ctx = model.encode(
    [context_text(problems[i]) for i in train_ids],
    batch_size=128,
    convert_to_tensor=True,
    show_progress_bar=True
)

print("Encoding all answers/contexts...")
all_ans = model.encode(
    [ans(problems[i]) for i in all_ids],
    batch_size=128,
    convert_to_tensor=True,
    show_progress_bar=True
)
all_ctx = model.encode(
    [context_text(problems[i]) for i in all_ids],
    batch_size=128,
    convert_to_tensor=True,
    show_progress_bar=True
)

print("Retrieving in batches...")
for start in tqdm(range(0, len(all_ids), 128)):
    end = min(start + 128, len(all_ids))
    score = (
    util.cos_sim(all_q[start:end], train_q)
    + util.cos_sim(all_ans[start:end], train_ans)
    + util.cos_sim(all_ctx[start:end], train_ctx)
    )

    for row, qid in enumerate(all_ids[start:end]):
        q_modality = all_modalities[start + row]
        modality_bonus = torch.tensor(
            [0.2 if tm == q_modality else 0.0 for tm in train_modalities],
            device=score.device
        )
        score[row] = score[row] + modality_bonus

    for row, qid in enumerate(all_ids[start:end]):
        topn = min(500, len(train_ids))
        _, idxs = torch.topk(score[row], k=topn)

        rel = []
        seen_questions = set()
        seen_answers = set()

        # pass 1: prefer unique question forms
        for idx in idxs.tolist():
            tid = train_ids[idx]
            if tid == qid:
                continue

            cand_question = question_signature(problems[tid].get("question", ""))
            cand_answer = normalize_text(ans(problems[tid]))

            if cand_question in seen_questions or cand_answer in seen_answers:
                continue

            rel.append(tid)
            seen_questions.add(cand_question)
            seen_answers.add(cand_answer)

            if len(rel) == K:
                break

        # pass 2: still prefer unique questions
        if len(rel) < K:
            for idx in idxs.tolist():
                tid = train_ids[idx]
                if tid == qid or tid in rel:
                    continue

                cand_question = question_signature(problems[tid].get("question", ""))
                cand_answer = normalize_text(ans(problems[tid]))

                if cand_question in seen_questions or cand_answer in seen_answers:
                    continue

                rel.append(tid)
                seen_questions.add(cand_question)
                seen_answers.add(cand_answer)

                if len(rel) == K:
                    break

        # pass 3: emergency fallback to guarantee exactly K
        if len(rel) < K:
            for idx in idxs.tolist():
                tid = train_ids[idx]
                if tid == qid or tid in rel:
                    continue

                rel.append(tid)

                if len(rel) == K:
                    break

        problems[qid]["relevant_question"] = rel

json.dump(problems, open(OUT, "w"), indent=2)
print("Saved", OUT)