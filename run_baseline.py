# from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch

model_name = "Qwen/Qwen-1_8B-Chat"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True
).eval()

query = "Answer: gravity\nPlease generate a question from the corresponding answer."
# query = "Picture: <img>"

print("Running inference...")
# response, history = model.chat(tokenizer, query=query, history=None)
inputs = tokenizer(query, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== MODEL OUTPUT ===")
print(response)
