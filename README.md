# Model Description: Chain-of-Exemplar for MCQ Generation
Option 2 - Improving upon Chain of Exemplar ([Luo et. al. 2024](https://aclanthology.org/2024.acl-long.432.pdf))

This project aims to address the lack of generated question diversity and mitigate hallucinations in the Chain of Exemplar model through in-context unsupervised learning, self-consistency, and potentially other methods that we come across throughout the quarter.

We are each working on a designated individual branch and will merge final aggregated changes at project milestones
- Josephine's branch: [https://github.com/SCU-CSEN346/Chain-of-Exemplar/tree/josephine]
- Tara's branch: [https://github.com/SCU-CSEN346/Chain-of-Exemplar/tree/tara]
- Kajal's branch: [https://github.com/SCU-CSEN346/Chain-of-Exemplar/tree/kajal]

## Dataset
We are using the ScienceQA dataset as modified by Luo et. al. for CoE: [CoE ScienceQA dataset](https://huggingface.co/datasets/Lhh123/CoE_ScienceQA)

---
# Member Contributions: 
### OVERALL PROJECT STATUS: 
- [x] Ran Inference on HPC with HF Transformers successfully with image + prompt as input and a generated question
- [x] Downloaded datasets and converted them to correct format for CoE
- [x] Ran full CoE baseline with the question, rationale, and distractor generations with evaluation metrics for 1 sample from test set
- [x] Run full CoE baseline with the question, rationale, and distractor generations with evaluation metrics to use as a comparison for our improvements
- [ ] Since our dataset will not go on HF, add instructions for finding it to the readme
- [ ] Improve this baseline with our own implementations

### JOSEPHINE - 33.3%: 
- [x] Set up project in HPC with VSCODE
- [x] Clone main branch to personal branch (Josephine)
- [x] Environment setup
- [x] Git setup including .gitignore (Commit + Push Git Setup)
- [x] Paper: Added GitHub + HF links to paper abstract and wrote In-Context Learning (2.2)
- [x] Got first baseline output by doing the OG repo's quickstart with HF transformers
- [x] Create Personal ReadMe with setup+inference documentation + Update Group ReadMe with documentation
- [x] Paper: Wrote Papers Methodology section in overleaf
- [x] Built ScienceQA → CoE pipeline (CER + multitask QG/RG/DG)
- [x] Created Slurm scripts for FULL CoE pipeline (training + inference + evaluation)
- [x] Finetuned LoRA model on the CoE multitask dataset
- [x] Ran full CoE pipeline on finetuned model with 1 sample from test set and gathered evaluation metrics
- [x] Ran full CoE pipeline on full test set: QG → RG → DG
- [x] Launched final scoring job for QG + RG + DG through Slurm
- [x] Pushed full test-set inference/evaluation scripts to GitHub
- [x] Paper: Wrote section 6.2 (Full CoE Pipeline - 1-Sample Result)
- [x] Update final README with baseline result
- [x] Paper: Wrote section 6 (Paper Results) and updated other sections to be up to date with advances in project.
- [x] Implement improvement - CER - attempt 1
- [x] Implement improvement - CER - attempt 2 (waiting for full pipeline to run to see results)
- [x] Paper: Wrote Section 7.1 (Improvement 1: Contextualized Exemplar Retrieval (CER))

### TARA - 33.3%: 
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

### KAJAL - 33.3%: 
- [x] First commit + Push of main papers github to this repo
- [x] Not using HPC, Local WSL + GPU environment setup completed
- [x] Completed the Abstract, Introduction, Chain-of-Thought Reasoning (2.3), and Multimodal Learning for Educational
NLP (2.5) subsections in the Related Work section in the overleaf document.
- [x] Fixed broken requirements.txt and resolved multiple dependency and version conflicts
- [x] Created run_baseline.py and successfully ran baseline GPU inference
- [x] Replaced unavailable original base model with Qwen/Qwen-1_8B-Chat
- [x] Verified baseline text generation with sample prompt-output pair
- [x] Added the installation instructions to the group ReadMe file 
- [x] Set up Hugging Face authentication and confirmed model downloading/inference workflow
- [x] Added reproducibility/setup documentation and .gitignore
- [x] Began setup of authors’ CoE ScienceQA dataset and confirmed local dataset structure (train/, val/, test/, captions.json)
- [x] Wrote Papers Methodology section in overleaf 
--- 
# Installation Instructions: List dependencies and how to install them.
⚠️ Note: The original repository ([Github Repo](https://github.com/Luohh5/Chain-of-Exemplar)) is not directly reproducible. See below for all fixes applied.

## 1. HPC Environment Setup

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
"numpy<2"
```

### 1.7 Verify versions

Expected:

```text
torch 2.0.1+cu118
transformers 4.32.0
peft 0.4.0
numpy 1.26.4
```

---
## 2. Git Setup for HPC

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
## 3. Quickstart Inference with Hugging Face Transformers
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

## 4. Slurm GPU Job

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

## 5. Example Successful Run

A successful run looked like this:

```text
torch 2.0.1+cu118
cuda? True
transformers 4.32.0
peft 0.4.0
What is the difference between the encoder and the decoder in a transformer?
```

That output is expected because the quickstart prompt asks the model to generate a question from the image. In this case, I uploaded a graph of what a encoder-decoder looks like.

---

# Usage Instructions: How to run the main script and reproduce the results.
Load the dataset - [Dataset](https://huggingface.co/datasets/Lhh123/CoE_ScienceQA)

Dataset setup has been started using the authors’ released CoE ScienceQA dataset. Full loading and preprocessing instructions will be added after baseline dataset integration is finalized.

In progress - will fill in once we are able to get a full CoE baseline

---
# Expected Output: Example results or sample output.
In progress - will fill in once we are able to get a baseline of the full CoE pipeline
