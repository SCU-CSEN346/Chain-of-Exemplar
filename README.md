# Josephine's ReadMe for Chain-of-Exemplar
## WHAT JOSEPHINE HAS DONE THUS FAR: 
- [x] Set up project in HPC with VSCODE
- [x] Clone main branch to personal branch (Josephine)
- [x] Environment setup
- [x] Git setup including .gitignore (Commit + Push Git Setup)
- [x] Paper: Added GitHub + HF links to paper abstract and wrote In-Context Learning (2.2)
- [x] Got first baseline output by doing the OG repo's quickstart with HF transformers
- [x] Create Personal ReadMe with setup+inference documentation + Update Group ReadMe with documentation
- [x] Paper: Wrote Papers Methodology section in overleaf
- [x] Downloaded and converted data to CoE format
- [x] Created a Slurm script (run_lora_train.slurm) to run training properly on GPU and started LoRA finetuning on the dataset

---

# 1. HPC Environment Setup

### 1.1 Connect to WAVE

```bash
ssh your_username@login.wave.scu.edu
```

### 1.2 Load Anaconda

```bash
module purge
module load Anaconda3
```

### 1.3 Create the conda environment in project storage

```bash
conda create --prefix /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu python=3.10 -y
```

You can activate it:

```bash
conda activate /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu
```

### 1.4 Create cache and temp folders

These avoid home-directory quota issues on the HPC.

```bash
mkdir -p /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/pip_cache
mkdir -p /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/tmp
mkdir -p /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache
mkdir -p /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/mpl_cache
mkdir -p /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/xdg_cache
```

### 1.5 Install PyTorch

Install the CUDA 11.8 build:

```bash
TMPDIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/tmp \
PIP_CACHE_DIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/pip_cache \
/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu/bin/python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### 1.6 Install required Python packages

```bash
TMPDIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/tmp \
PIP_CACHE_DIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/pip_cache \
/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu/bin/python -m pip install \
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
deepspeed==0.9.5 \
"numpy<2"
```

DeepSpeed is required because `finetune.py` imports it, even when running LoRA training.

### 1.7 Verify versions

Expected:

```text
torch 2.0.1+cu118
transformers 4.32.0
peft 0.4.0
numpy 1.26.4
```

---
# 2. Git Setup for HPC

### 2.1. Step 1 - Create `.gitignore`

Create or edit `.gitignore` (e.g. `nano .gitignore`) and add:

```gitignore
# Virtual environments
venv/
reproduction/Chain-of-Exemplar/venv/

# Python cache
__pycache__/
*.pyc

# Logs and env files
*.log
.env

# Data / outputs
outputs/
data/
.cache/

# Model files
*.pt
*.bin
```

In **nano**: save with `Ctrl + O` → Enter → exit with `Ctrl + X`.

### 2.2. Step 2 - Commit `.gitignore`

```bash
git add .gitignore
git commit -m "add gitignore to ignore venv"
git push
```
---
# 3. Quickstart Inference with Hugging Face Transformers
This part explains how to run our baseline of Chain-of-Exemplar inference of their paper on the WAVE HPC cluster using the Hugging Face Transformers quickstart. This does not yet include the full CoE pipeline.

### Important notes:
- Use `Qwen/Qwen-VL-Chat`
- Do **not** use `Qwen/Qwen-VL-Chat-Int4` for this setup
- This setup runs inference with the adapter model: `Lhh123/coe_multitask_blip2xl_angle_2ep`
- The quickstart does **not** require datasets but will once the full CoE pipeline is run


### 3.1 Download the adapter model

Do not use plain git clone from Hugging Face unless Git LFS is available and working.

Use snapshot_download instead:

```bash
HF_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache \
HUGGINGFACE_HUB_CACHE=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache/hub \
XDG_CACHE_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/xdg_cache \
TMPDIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/tmp \
/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Lhh123/coe_multitask_blip2xl_angle_2ep",
    local_dir="/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_multitask_blip2xl_angle_2ep",
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

### 3.2 Fix the adapter config

Edit:
```bash
nano /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_multitask_blip2xl_angle_2ep/adapter_config.json
```

Set:
```json
"base_model_name_or_path": "Qwen/Qwen-VL-Chat"
```

This is the final working base model.

### 3.3 Add a test image

Use a .jpg, .jpeg, or .png image and add it to your project folder with name test.jpg. In our case that was:

```bash
/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar/test.jpg
```

### 3.4 Create or use run_inference.py

At this step, you can either:

- use the run_inference.py file already pushed to GitHub (Josephine's branch), or
- create the file yourself as described below

If you want to create it yourself:
```bash
cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar
nano run_inference.py
```

Use:
```python
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

IMAGE_PATH = "/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar/test.jpg"
MODEL_PATH = "/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_multitask_blip2xl_angle_2ep"

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

query = f"Picture: <img>{IMAGE_PATH}</img>\nGenerate a question based on the picture."

response, history = model.chat(
    tokenizer,
    query=query,
    history=None
)

print(response)
```

This prompt asks the model to generate a question from the image.

---

# 4. Slurm GPU Job

### 4.1 Create or use run_coe.slurm

At this step, you can either:

- use the run_coe.slurm file already pushed to GitHub (Josephine's branch), or
- create the file yourself as described below

If you want to create it yourself:
```bash
cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar
nano run_coe.slurm
```

Use:
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

module load Anaconda3

export HF_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache
export TRANSFORMERS_CACHE=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache
export HUGGINGFACE_HUB_CACHE=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache/hub
export MPLCONFIGDIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/mpl_cache
export XDG_CACHE_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/xdg_cache
export PIP_CACHE_DIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/pip_cache
export TMPDIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/tmp

mkdir -p $HF_HOME
mkdir -p $HUGGINGFACE_HUB_CACHE
mkdir -p $MPLCONFIGDIR
mkdir -p $XDG_CACHE_HOME
mkdir -p $PIP_CACHE_DIR
mkdir -p $TMPDIR

cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu/bin/python --version

/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu/bin/python -c "import torch, transformers, peft; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available()); print('transformers', transformers.__version__); print('peft', peft.__version__)"

/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu/bin/python run_inference.py
```

The Slurm script already handles environment setup, including clearing conflicting Python paths and setting cache directories. Do not run training on the login node.

Important: `#SBATCH --gres=gpu:volta:1` is required so Slurm allocates a real GPU.

### 4.2 Run the job

Submit:
```bash
sbatch run_coe.slurm
```

Check status:
```bash
squeue -u $USER
```

After it finishes:
```bash
cat coe_infer_<JOBID>.out
cat coe_infer_<JOBID>.err
```

---

# 5. Example Successful Run

A successful run looked like this:

```text
torch 2.0.1+cu118
cuda? True
transformers 4.32.0
peft 0.4.0
What is the difference between the encoder and the decoder in a transformer?
```

That output is expected because the quickstart prompt asks the model to generate a question from the image. In this case, I uploaded a graph of what a encoder-decoder looks like.

# 6. Troubleshooting

### Wrong Python packages are being used

If you see errors like:

```text
ModuleNotFoundError: peft
ModuleNotFoundError: torch
```

make sure you are using:

```text
/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu/bin/python
```

### cuda? False

Slurm did not allocate a GPU. Make sure the Slurm file includes:

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
```

### Quota errors

If pip or Hugging Face tries to write into your home directory, make sure these are set:

```bash
export HF_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache
export TRANSFORMERS_CACHE=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache
export HUGGINGFACE_HUB_CACHE=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache/hub
export XDG_CACHE_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/xdg_cache
export PIP_CACHE_DIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/pip_cache
export TMPDIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/tmp
```

### NumPy errors

If you see NumPy compatibility errors, use:

```bash
/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu/bin/python -m pip install "numpy<2"
```

numpy 1.26.4 worked.

### _pickle.UnpicklingError: invalid load key, 'v'

This means adapter_model.bin is a Git LFS pointer file, not the real model. Use snapshot_download instead of plain git clone.

### Garbage output with Qwen/Qwen-VL-Chat-Int4

The Int4 model did not load correctly in this environment. Use:

```json
"base_model_name_or_path": "Qwen/Qwen-VL-Chat"
```

### Do not install latest bitsandbytes

Latest bitsandbytes upgraded torch to a CUDA 13 build and broke compatibility with the Tesla V100 setup.

---

# 7. Dataset Preparation, LoRA Finetuning, and End-to-End Pipeline

This section explains the full pipeline on WAVE HPC from dataset preparation to training to inference.

## 7.1 Navigate to the correct repo folder

All commands should be run from:

```bash
cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar
```

IMPORTANT: All commands must be run from the `reproduction/Chain-of-Exemplar` folder. This is the only folder that contains `finetune.py`. Running from the wrong directory will result in file-not-found errors.

## 7.2 Activate the environment

```bash
conda activate /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu
```

## 7.3 Dataset Preparation

We downloaded the ScienceQA dataset using the Hugging Face `datasets` library and saved it locally to project storage on the HPC.

The dataset was then converted into the Chain-of-Exemplar conversation format. Each sample is stored as a dictionary with:

- an "id"
- a list of "conversations"

Each conversation follows:

```json
{
  "from": "user",
  "value": "Picture: <img>path/to/image.png</img>\nAnswer: ...\nPlease generate a question..."
},
{
  "from": "assistant",
  "value": "..."
}
```

Images from ScienceQA are stored locally under:

```text
data/images/train/
data/images/validation/
data/images/test/
```

and referenced in the JSON using `<img>path</img>` tokens.

The conversion script used is (on Josephine's branch):

```text
convert_all_splits_to_coe.py
```

This script produces:

```text
data/coe_train.json
data/coe_validation.json
data/coe_test.json
```

The dataset is generated using:

```bash
python convert_all_splits_to_coe.py
```

This script automatically downloads ScienceQA using the Hugging Face datasets library and converts it into the Chain-of-Exemplar format. No manual dataset download is required.

Note: These dataset files and images are not pushed to GitHub because they are large and are ignored via `.gitignore`.

## 7.4 Run LoRA Finetuning (Slurm)

We finetune the Qwen-VL-Chat model using LoRA on the prepared dataset.

Training is launched using the Slurm script (on Josephine's branch):

```text
run_lora_train.slurm
```

This Slurm script is already configured for the WAVE HPC environment, including GPU allocation, cache paths, and environment settings.

This script runs:

- Base model: `Qwen/Qwen-VL-Chat`
- Dataset: `data/coe_train.json`
- Method: LoRA finetuning
- Precision: `fp16` (`bf16` is not supported on Tesla V100)
- Tesla V100 GPUs do NOT support `bf16`. Using `bf16` will crash with a `ValueError`. Always use `fp16`.
- Output directory: `output_lora_coe_train/`

Submit the training job:

```bash
sbatch run_lora_train.slurm
```

Check job status:

```bash
squeue -u $USER
```

View logs:

```bash
cat coe_lora_<JOBID>.out
cat coe_lora_<JOBID>.err
```

Training may take several hours depending on GPU availability.
Initial model loading may take several minutes with no output - this is expected.

## 7.5 Run Inference with Finetuned Model

After training completes, locate the output directory:

```text
output_lora_coe_train/
```

Create a new inference script:

```bash
nano run_inference_lora.py
```

Use:

```python
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

MODEL_PATH = "output_lora_coe_train"
IMAGE_PATH = "test.jpg"

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

query = f"Picture: <img>{IMAGE_PATH}</img>\nGenerate a question based on the picture."

response, history = model.chat(
    tokenizer,
    query=query,
    history=None
)

print(response)
```

Run it:

```bash
/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu/bin/python run_inference_lora.py
```

After training completes, the output directory contains LoRA adapter weights that can be used for inference.
