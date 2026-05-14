import json, os, torch
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

MODEL_PATH = "output/fullcoe_lora_v100_2ep_retrievalfix"
DATA_PATH = "data/ScienceQA_test_rg_from_qg_2ep.json"
OUT_PATH = "infer/pred_test_rg_2ep.json"

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
    response, _ = model.chat(tokenizer, query=prompt, history=None)
    preds.append({
        "id": ex["id"],
        "response": response
    })

json.dump(preds, open(OUT_PATH, "w"), indent=2)
print("saved", OUT_PATH)
print("samples", len(preds))
print(preds[0])
