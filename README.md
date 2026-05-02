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
- [x] Built ScienceQA → CoE pipeline (CER + multitask QG/RG/DG)
- [x] Created Slurm scripts for FULL CoE pipeline (slurm_fullcoe_lora.sbatch) to run training properly on GPU and started LoRA finetuning on the dataset

---

## ⚠️ IMPORTANT

This README contains two workflows:

- Sections 3–5: Quickstart inference (sanity check only)
- Section 7: FULL Chain-of-Exemplar pipeline (used for reproduction)

If your goal is to reproduce the paper, go directly to Section 7.

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

# 7. Full Chain-of-Exemplar Pipeline (Correct Baseline)

This section describes the FULL CoE pipeline:

1. ScienceQA dataset reconstruction
2. CER (Contextualized Exemplar Retrieval)
3. Multitask training (QG + RG + DG)
4. LoRA finetuning
5. (Later) inference pipeline

---

## 7.1 Navigate to repo

```bash
cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar
conda activate /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu
```

## 7.2 Install extra packages for FULL CoE pipeline

The full CoE pipeline needs extra packages beyond the quickstart inference setup.

Run this inside the `coe_gpu` environment:

```bash
conda activate /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu
```
Set cache directories first to avoid HPC home-directory quota errors:
```bash
export HF_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export SENTENCE_TRANSFORMERS_HOME=$HF_HOME/sentence_transformers
export PIP_CACHE_DIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/pip_cache
export TMPDIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/tmp

mkdir -p $HF_HOME $HUGGINGFACE_HUB_CACHE $SENTENCE_TRANSFORMERS_HOME $PIP_CACHE_DIR $TMPDIR
```
Install packages needed for ScienceQA, CER, and multitask preparation:
```bash
pip install datasets pillow jsonlines requests==2.32.3
pip install huggingface-hub==0.25.2
pip install sentence-transformers==2.2.2
```
Important: do not install the latest sentence-transformers, because it upgrades transformers and can break the Qwen-VL setup.

## 7.3 Build ScienceQA dataset

```bash
python build_scienceqa_problems.py
```

Creates:

```text
data/scienceqa/problems.json
data/scienceqa/problems_blip2xl_angle.json
```

## 7.4 Run CER (Contextualized Exemplar Retrieval)

```bash
python retrieve.py
```
Note: This uses a modified `retrieve.py` that replaces the original AnglE embedding with sentence-transformers for compatibility on HPC.

Adds:

```json
"relevant_question": [...]
```

to each sample.

## 7.5 Build multitask dataset
```bash
python prepare_multitask.py
```

Creates:

```text
data/ScienceQA_train_multitask_fullcoe.json
```

This dataset includes:

QG (Question Generation)
RG (Rationale Generation)
DG (Distractor Generation)

## 7.6 Train model (LoRA)

Submit:
```bash
sbatch slurm_fullcoe_lora.sbatch
```
Config:

- Model: `Qwen/Qwen-VL-Chat`
- Method: LoRA
- Precision: `fp16`
- Epochs: 2
- GPU: Tesla V100

Expected runtime:

```text
6–12 hours
```

## 7.7 Output
```text
output/fullcoe_lora_v100/
```
Contains:
```text
adapter_model.bin
checkpoint-*
```

---

