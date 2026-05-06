# Tara's README for Chain-of-Exemplar

## My Contributions Thus Far:
- [x] Set up project in HPC
- [x] Set up conda environment + documenting setup locally
- [x] Working on getting baseline to work on HPC
- [x] Model description in Group ReadMe
- [x] Wrote Automatic Multiple Choice Question Generation (2.1) and Self-Consistency (2.4) subsections in Related Works section in paper
- [x] Wrote Datasets section (3) in paper
- [x] Tried a different way to run the paper's quickstart inference in the HPC due to package conflicts/updates
- [x] Following Josephine's setup to get inference to work on HPC
- [x] Figured out how to run inference on HPC using simpler Conda setup
- [x] Fixed Conda setup to work for LoRA fine-tuning on HPC
- [x] Have baseline model running on HPC
- [x] Wrote Paper Evaluation section (5) in paper

## How to Run Inference on HPC

### 1. Create a new conda environment
```bash
module purge
module load Anaconda3
conda create -n coe python=3.10 -y
conda activate coe
```

### 2. Install dependencies

```bash
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python -m pip install \
  transformers==4.32.0 \
  peft==0.4.0 \
  accelerate==0.21.0 \
  datasets \
  tqdm \
  sentencepiece \
  einops \
  matplotlib \
  tiktoken \
  transformers_stream_generator \
  "numpy<2"
```

***Note: Don't install the original project's requirements.txt. It contains far more packages that will max out space in your home directory upon installation to the conda environment. Above is the minimum for running inference.***

### 3. Download pretrained COE model
#### Use snapshot download to download Luo et. al.'s pretrained COE model from HuggingFace: [https://huggingface.co/Lhh123/coe_multitask_blip2xl_angle_2ep]
```bash
cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Lhh123/coe_multitask_blip2xl_angle_2ep",
    local_dir="/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/coe_multitask_blip2xl_angle_2ep",
    allow_patterns=[
        "adapter_config.json",
        "adapter_model.bin",
        "README.md",
        "qwen.tiktoken",
        "special_tokens_map.json",
        "tokenization_qwen.py",
        "tokenizer_config.json",
    ],
)
PY
```
#### Change base model adapter from Qwen-VL-Chat-Int4 to Qwen/Qwen-VL-Chat
In .../coe_multitask_blip2xl_angle_2ep/adapter_config.json, set:
```JSON
"base_model_name_or_path": "Qwen/Qwen-VL-Chat"
```

### 4. Add a test image to working directory
Use a .jpg, .jpeg, or .png image and add it to your project folder with name test.jpg. In our case that was:
```bash
mkdir images/testset/transformer_architecture_img.png
```

### 5. Create `run_inference.py`

```python
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

IMAGE_PATH = "/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/Chain-of-Exemplar/reproduction/Chain-of-Exemplar/images/testset/transformer_architecture_img.png"
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

query = f"Picture: <img>{IMAGE_PATH}</img>\n\n Generate a question based on the picture."

response, history = model.chat(
    tokenizer,
    query=query,
    history=None
)

print(response)
```

### 6. Create SLURM script

```bash
#!/bin/bash
#SBATCH --job-name=coe_infer
#SBATCH --output=coe_infer_%j.out
#SBATCH --error=coe_infer_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=tkhambadkone@scu.edu
#SBATCH --mail-type=END,FAIL

module purge
module load Anaconda3
module load CUDA/12.2.1
conda activate coe

/WAVE/users2/unix/tkhambadkone/.conda/envs/coe/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'is_available?', torch.cuda.is_available())"

cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/Chain-of-Exemplar/reproduction/Chain-of-Exemplar
/WAVE/users2/unix/tkhambadkone/.conda/envs/coe/bin/python run_inference.py
```

### 7. Run SLURM script

```bash
sbatch run_coe.slurm
```
See inference results in file `coe_infer_<BATCH_JOB_ID>.out` and error log in file `coe_infer_<BATCH_JOB_ID>.err`
