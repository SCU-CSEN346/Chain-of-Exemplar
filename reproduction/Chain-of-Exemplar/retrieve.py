import json, os, torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

DATA_ROOT = "data/scienceqa"
INP = f"{DATA_ROOT}/problems.json"
OUT = f"{DATA_ROOT}/problems_blip2xl_angle.json"
K = 5
MODEL_NAME = "all-MiniLM-L6-v2"

problems = json.load(open(INP))
train_ids = [k for k,v in problems.items() if v["split"] == "train"]
all_ids = list(problems.keys())

def ans(p): return p["choices"][p["answer"]]
def ques(p): return p["question"]

model = SentenceTransformer(MODEL_NAME)

print("Encoding train answers/questions...")
train_ans = model.encode([ans(problems[i]) for i in train_ids], batch_size=128, convert_to_tensor=True, show_progress_bar=True)
train_q = model.encode([ques(problems[i]) for i in train_ids], batch_size=128, convert_to_tensor=True, show_progress_bar=True)

print("Encoding all answers/questions...")
all_ans = model.encode([ans(problems[i]) for i in all_ids], batch_size=128, convert_to_tensor=True, show_progress_bar=True)
all_q = model.encode([ques(problems[i]) for i in all_ids], batch_size=128, convert_to_tensor=True, show_progress_bar=True)

print("Retrieving in batches...")
for start in tqdm(range(0, len(all_ids), 128)):
    end = min(start + 128, len(all_ids))
    score = util.cos_sim(all_ans[start:end], train_ans) + util.cos_sim(all_q[start:end], train_q)

    for row, qid in enumerate(all_ids[start:end]):
        vals, idxs = torch.topk(score[row], k=K+1)
        rel = []
        for idx in idxs.tolist():
            tid = train_ids[idx]
            if tid != qid:
                rel.append(tid)
            if len(rel) == K:
                break
        problems[qid]["relevant_question"] = rel

json.dump(problems, open(OUT, "w"), indent=2)
print("Saved", OUT)