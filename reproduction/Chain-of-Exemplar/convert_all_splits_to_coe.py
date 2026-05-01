import os
import json
from tqdm import tqdm
from datasets import load_from_disk

ds = load_from_disk("data/scienceqa")

for split in ["train", "validation", "test"]:
    image_dir = f"data/images/{split}"
    output_path = f"data/coe_{split}.json"
    os.makedirs(image_dir, exist_ok=True)

    output = []

    for i, sample in enumerate(tqdm(ds[split], desc=split)):
        img = sample["image"]

        answer_idx = sample["answer"]
        answer_text = sample["choices"][answer_idx]

        if img is not None:
            img_path = f"{image_dir}/{i}.png"
            img.save(img_path)

            user_prompt = (
                f"Picture: <img>{img_path}</img>\n"
                f"Answer: {answer_text}\n"
                f"Please generate a question from this picture and the corresponding answer."
            )
        else:
            user_prompt = (
                f"Answer: {answer_text}\n"
                f"Please generate a question from the corresponding answer."
            )

        output.append({
            "id": f"{split}_{i}",
            "conversations": [
                {"from": "user", "value": user_prompt},
                {"from": "assistant", "value": sample["question"]}
            ]
        })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {output_path}: {len(output)} samples")
