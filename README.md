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

- Section 3: Quickstart inference (sanity check only)
- Sections 5–6: FULL Chain-of-Exemplar pipeline (used for reproduction)

If your goal is to reproduce the paper, go directly to Section 5.

## Repository Structure

Key scripts:

- `run_inference.py` → quickstart inference
- `run_coe.slurm` → inference job
- `build_scienceqa_problems.py` → dataset builder
- `retrieve.py` → CER
- `prepare_multitask.py` → multitask dataset
- `slurm_fullcoe_lora.sbatch` → training job
- `run_fullcoe_inference.py` → full CoE inference

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
# 3. Quickstart Inference with HF Transformers (Sanity Check)

This section runs a minimal inference example to verify the environment and model setup.

This uses scripts already included in the repository.

---

## 3.1 Required files

Ensure the following files exist:

```text
run_inference.py
run_coe.slurm
```

If not, pull the latest version:

```bash
git pull
```

---

## 3.2 Download adapter model

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Lhh123/coe_multitask_blip2xl_angle_2ep",
    local_dir="coe_multitask_blip2xl_angle_2ep"
)
PY
```

---

## 3.3 Fix adapter config

Edit:

```bash
nano coe_multitask_blip2xl_angle_2ep/adapter_config.json
```

Set:

```json
"base_model_name_or_path": "Qwen/Qwen-VL-Chat"
```

---

## 3.4 Add test image

Place an image at:

```text
test.jpg
```

---

## 3.5 Run inference (Slurm)

```bash
sbatch run_coe.slurm
```

Check job:

```bash
squeue -u $USER
```

View output:

```bash
cat coe_infer_<JOBID>.out
```

---

## 3.6 Expected result

Example output:

```text
What is the difference between the encoder and the decoder in a transformer?
```

This confirms:
- GPU works
- model loads
- inference pipeline works

# 4. Troubleshooting

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

# 5. Full Chain-of-Exemplar Pipeline (Correct Baseline)

This section describes the FULL CoE pipeline:

1. ScienceQA dataset reconstruction  
2. CER (Contextualized Exemplar Retrieval)  
3. Multitask data construction (QG + RG + DG)  
4. LoRA finetuning  
5. Inference pipeline (Section 6) 

---

## 5.1 Navigate to repo

```bash
cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar
conda activate /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu
```

## 5.2 Install extra packages for FULL CoE pipeline

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

### Verify environment (IMPORTANT)

Run:

```bash
python - <<'PY'
import torch, transformers, huggingface_hub, datasets, sentence_transformers
from transformers import Trainer

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("huggingface-hub:", huggingface_hub.__version__)
print("datasets:", datasets.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
print("Trainer import OK")
PY
```

Expected:

```text
torch: 2.0.1+cu118
transformers: 4.32.0
huggingface-hub: 0.25.2
sentence-transformers: 2.2.2
Trainer import OK
```

## 5.3 Required scripts (IMPORTANT)

The full CoE pipeline depends on custom scripts added in this repository.

Make sure the following files exist in:

```text
reproduction/Chain-of-Exemplar/
```

Required scripts:

```text
build_scienceqa_problems.py
retrieve.py
prepare_multitask.py
finetune/run_fullcoe_lora_v100.sh
slurm_fullcoe_lora.sbatch
```

If you cloned this repository, these should already be present.  
If not, pull the latest version:

```bash
git pull
```

## 5.4 Build ScienceQA dataset

```bash
python build_scienceqa_problems.py
```

Creates:

```text
data/scienceqa/problems.json
data/scienceqa/problems_blip2xl_angle.json
```

## 5.5 Run CER (Contextualized Exemplar Retrieval)

```bash
python retrieve.py
```

Note:
- This uses a modified `retrieve.py` that replaces the original AnglE embedding with sentence-transformers for compatibility on HPC.
- CER may take 10–30 minutes depending on compute.

Adds:

```json
"relevant_question": [...]
```

to each sample.

## 5.6 Build multitask dataset

```bash
python prepare_multitask.py
```

Creates:

```text
data/ScienceQA_train_multitask_fullcoe.json
```

Expected size:

```text
~38,000 samples (3× original train set)
```

This dataset includes:

- QG (Question Generation)
- RG (Rationale Generation)
- DG (Distractor Generation)

## 5.7 Train model (LoRA)

Submit:
```bash
sbatch slurm_fullcoe_lora.sbatch
```
Config:

- Model: `Qwen/Qwen-VL-Chat`
- Method: LoRA
- Precision: `fp16`
- Epochs: 1
- GPU: Tesla V100

Expected runtime:

```text
12–14 hours
```

## 5.8 Output
```text
output/fullcoe_lora_v100_1ep/
```
Contains:
```text
adapter_model.bin
checkpoint-*
```
### Shared trained model on WAVE

The trained 1-epoch and 2-epoch LoRA adapter is not stored in GitHub because it is too large.

On WAVE, group members can access it at:

```text
1 epoch: /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar/output/fullcoe_lora_v100_1ep
2 epoch: /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar/output/fullcoe_lora_v100_2ep
```
---

# 6. Full CoE Inference Pipeline (QG → RG → DG) 

After training, the Chain-of-Exemplar pipeline requires **chaining three tasks**:

```text
Question Generation (QG) → Rationale Generation (RG) → Distractor Generation (DG)
```

This section describes how to run full inference using the trained model.

---

## 6.1 Create inference script

Create a new file:

```bash
nano run_fullcoe_inference.py
```

Paste the following:

```python
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

MODEL_PATH = "output/fullcoe_lora_v100_1ep"

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

def ask(prompt):
    response, _ = model.chat(
        tokenizer,
        query=prompt,
        history=None
    )
    return response

# Example input
IMAGE_PATH = "data/images/test.jpg"

# Step 1: QG (generate question)
qg_prompt = f"Picture: <img>{IMAGE_PATH}</img>\nGenerate a question based on the picture."
question = ask(qg_prompt)
print("QG:", question)

# Step 2: RG (generate reasoning)
rg_prompt = f"Picture: <img>{IMAGE_PATH}</img>\nQuestion: {question}\nAnswer: (your answer here)\nExplain the reasoning."
reasoning = ask(rg_prompt)
print("RG:", reasoning)

# Step 3: DG (generate distractors)
dg_prompt = f"Picture: <img>{IMAGE_PATH}</img>\nQuestion: {question}\nAnswer: (your answer here)\nGenerate distractors."
distractors = ask(dg_prompt)
print("DG:", distractors)
```

---

## 6.2 Run inference

```bash
python run_fullcoe_inference.py
```

---

## 6.3 Notes

- You must manually provide the **correct answer** for RG and DG steps.
- The model was trained multitask, so it understands all three prompts.
- Each step depends on the previous output.

---

## 6.4 Important

This chaining step is **required** to reproduce the behavior of the Chain-of-Exemplar paper.

Training alone is not sufficient — inference must follow:

```text
QG → RG → DG
```
