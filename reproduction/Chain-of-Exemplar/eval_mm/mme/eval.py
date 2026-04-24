import os
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoTokenizer

# base_model = AutoPeftModelForCausalLM.from_pretrained("Qwen-VL-Chat")
# model = AutoPeftModelForCausalLM.from_pretrained(base_model, 
#         "Lhh123/coe_multitask_blip2xl_angle_2ep",
#         device_map="cuda",
#         trust_remote_code=True).eval()

# checkpoint = 'Qwen/Qwen-VL-Chat' # path to the base model

MODEL_PATH = "/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/coe_multitask_blip2xl_angle_2ep"

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, MODEL_PATH).from_pretrained( # path to the output directory
    device_map="cuda",
    trust_remote_code=True
).eval()

# model.generation_config = GenerationConfig.from_pretrained(checkpoint, trust_remote_code=True)
# model.generation_config.top_p = 0.01


# root = 'Your_Results'
# output = 'Qwen-VL-Chat'
# os.makedirs(output, exist_ok=True)
# for filename in os.listdir(root):
#     # with open(os.path.join(root, filename), 'r') as fin, open(os.path.join(output, filename), 'w') as fout:
#     #     lines = fin.read().splitlines()
#     #     filename = filename.replace('.txt', '')
#     #     for line in tqdm(lines):
#     #         img, question, gt = line.strip().split('\t')
#     #         img_path = os.path.join('images', filename, img)
#     #         assert os.path.exists(img_path), img_path
query = "Picture: <img>/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/Chain-of-Exemplar/reproduction/Chain-of-Exemplar/images/testset/transformer_architecture_img.png</img>\nGenerate a question based on the picture."
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
