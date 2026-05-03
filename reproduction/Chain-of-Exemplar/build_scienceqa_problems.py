import os, json
from datasets import load_dataset

OUT_DIR = "data/scienceqa"
IMG_ROOT = "data/images"
os.makedirs(OUT_DIR, exist_ok=True)

ds = load_dataset("derek-thomas/ScienceQA")
problems = {}

for split in ["train", "validation", "test"]:
    for i, ex in enumerate(ds[split]):
        qid = f"{split}_{i}"

        image_path = None
        img = ex.get("image")
        if img is not None:
            os.makedirs(f"{IMG_ROOT}/{split}", exist_ok=True)
            image_path = f"{IMG_ROOT}/{split}/{i}.png"
            if not os.path.exists(image_path):
                img.save(image_path)

        problems[qid] = {
            "question": ex.get("question", ""),
            "choices": ex.get("choices", []),
            "answer": int(ex.get("answer", 0)),
            "hint": ex.get("hint", "") or "",
            "lecture": ex.get("lecture", "") or "",
            "solution": ex.get("solution", "") or "",
            "image": image_path,
            "split": split,
            "relevant_question": []
        }

json.dump(problems, open(f"{OUT_DIR}/problems.json", "w"), indent=2)
json.dump(problems, open(f"{OUT_DIR}/problems_blip2xl_angle.json", "w"), indent=2)

print("wrote", f"{OUT_DIR}/problems.json")
print("samples:", len(problems))
k = next(iter(problems))
print(k, problems[k])