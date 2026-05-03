import json, os

PROBLEMS = "data/scienceqa/problems_blip2xl_angle.json"
OUT = "data/ScienceQA_train_multitask_fullcoe.json"

def img_line(p):
    return f"Picture: <img>{p['image']}</img>\n" if p.get("image") else ""

def ctx_line(p):
    return f"Context: {p['hint']}\n" if p.get("hint") else ""

def ans_line(p):
    return f"Answer: {p['choices'][p['answer']]}\n"

def rationale(p):
    return (p.get("lecture") or "") + "\n" + (p.get("solution") or "")

def distractors(p):
    correct = p["choices"][p["answer"]]
    wrong = [c for c in p["choices"] if c != correct]
    return "".join(f"({i+1}) {c}\n" for i,c in enumerate(wrong))

def exemplar(p):
    base = img_line(p) + ctx_line(p)
    return {
        "qg": f"\nExample:\n{base}{ans_line(p)}Question: {p['question']}",
        "rg": f"\nExample:\n{base}Question: {p['question']}\n{ans_line(p)}Reasoning: {rationale(p)}",
        "dg": f"\nExample:\n{base}Question: {p['question']}\n{ans_line(p)}Distractors: {distractors(p)}",
    }

problems = json.load(open(PROBLEMS))
save = []

for qid, p in problems.items():
    if p["split"] != "train":
        continue

    base = img_line(p) + ctx_line(p)
    ans = ans_line(p)
    q = p["question"]
    rat = rationale(p)
    dis = distractors(p)

    ex = {"qg": "", "rg": "", "dg": ""}
    prefix = ""
    rel = p.get("relevant_question") or []
    if rel and rel[0] in problems:
        ex = exemplar(problems[rel[0]])
        prefix = "Refer to the following example, "

    save.extend([
        {
            "id": f"qg_{qid}",
            "conversations": [
                {"from": "user", "value": base + ans + prefix + "generate a question based on the above information and the corresponding answer." + ex["qg"]},
                {"from": "assistant", "value": q},
            ],
        },
        {
            "id": f"rg_{qid}",
            "conversations": [
                {"from": "user", "value": base + f"Question: {q}\n" + ans + prefix + "how to make a reasoning to answer the question based on the above information, question, and answer?" + ex["rg"]},
                {"from": "assistant", "value": rat},
            ],
        },
        {
            "id": f"dg_{qid}",
            "conversations": [
                {"from": "user", "value": base + f"Question: {q}\n" + ans + prefix + "generate plausible yet incorrect distractors similar to the correct answer and separate them with numbers like (1) (2) (3)." + ex["dg"]},
                {"from": "assistant", "value": dis},
            ],
        },
    ])

json.dump(save, open(OUT, "w"), indent=2)
print("saved", OUT)
print("samples", len(save))
print(save[0])