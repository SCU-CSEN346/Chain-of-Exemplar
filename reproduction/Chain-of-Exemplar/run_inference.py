import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

IMAGE_PATH = "/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/Chain-of-Exemplar/reproduction/Chain-of-Exemplar/images/testset/4105.png"
MODEL_PATH = "/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/coe_multitask_blip2xl_angle_2ep"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
).eval()

query = f"Picture: <img>{IMAGE_PATH}</img>\n Generate a question based on the picture."

response, history = model.chat(
    tokenizer,
    query=query,
    history=None
)

print(response)