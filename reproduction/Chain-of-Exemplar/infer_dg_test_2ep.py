import json, os, torch
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import re

MODEL_PATH = "output/fullcoe_lora_v100_2ep_retrievalfix"
DATA_PATH = "data/ScienceQA_test_dg_from_qg_rg_2ep_self_consistent_3samples_3distractors_final.json"
OUT_PATH = "infer/pred_test_dg_2ep_self_consistent_3samples_3distractors_final.json"

os.makedirs("infer", exist_ok=True)

data = json.load(open(DATA_PATH))

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).eval()

preds = []



for ex in tqdm(data):
    prompt = ex["conversations"][0]["value"]
    # Check for image path in prompt (Qwen-VL-Chat expects <img>path</img> tokens)
    def replace_img_url_with_local(match):
            url = match.group(1)
            parts = url.split('/')
            if len(parts) >= 2:
                subdir = parts[-2]
                filename = parts[-1]
                local_path = f"data/images/{subdir}/{filename}"
            else:
                local_path = f"data/images/{parts[-1]}"
            return f"<img>{local_path}</img>"
    
    #img_paths = re.findall(r'<img>(.*?)</img>', prompt)
    prompt = re.sub(r"<img>(.*?)</img>", replace_img_url_with_local, prompt)
    missing = False
    # for img_path in img_paths:
    #     if not os.path.exists(img_path):
    #         print(f"[WARNING] Skipping id {ex['id']}: missing image {img_path}")
    #         missing = True
    #         break
    if missing:
        continue
    try:
        response, _ = model.chat(tokenizer, query=prompt, history=None)
    except FileNotFoundError as e:
        print(f"[ERROR] Skipping id {ex['id']}: {e}")
        continue
    preds.append({
        "id": ex["id"],
        "response": response
    })

json.dump(preds, open(OUT_PATH, "w"), indent=2)
print("saved", OUT_PATH)
print("samples", len(preds))
print(preds[0])
