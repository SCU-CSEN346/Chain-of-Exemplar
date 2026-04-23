import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

IMAGE_PATH = "/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar/test.jpg"

tokenizer = AutoTokenizer.from_pretrained(
    "/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_multitask_blip2xl_angle_2ep",
    trust_remote_code=True
)

model = AutoPeftModelForCausalLM.from_pretrained(
    "/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_multitask_blip2xl_angle_2ep",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
).eval()

query = f"Picture: <img>{IMAGE_PATH}</img>\nGenerate a question based on the picture."

response, history = model.chat(
    tokenizer,
    query=query,
    history=None
)

print(response)
