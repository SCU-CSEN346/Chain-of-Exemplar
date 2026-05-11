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
- [ ] Remove any files on github that no longer are being used
- [ ] Improve Readme for personal branch and main branch

---

## ⚠️ IMPORTANT

This README contains two workflows:

- Section 3: Quickstart inference (sanity check only)
- Sections 5–8: FULL Chain-of-Exemplar pipeline (used for reproduction)

If your goal is to reproduce the paper, go directly to Section 5.


## Repository Structure

Key scripts:

- `run_inference.py` → quickstart inference
- `run_coe.slurm` → quickstart Slurm job
- `build_scienceqa_problems.py` → ScienceQA dataset builder
- `retrieve.py` → CER
- `prepare_multitask.py` → multitask dataset builder
- `slurm_fullcoe_lora.sbatch` → LoRA training job

Full test-set pipeline:
- `infer_qg_test_2ep.py` → full-test QG inference
- `prepare_rg_test_2ep.py` → build RG full-test input from QG outputs
- `infer_rg_test_2ep.py` → full-test RG inference
- `prepare_dg_test_2ep.py` → build DG full-test input from QG + RG outputs
- `infer_dg_test_2ep.py` → full-test DG inference
- `scoring_local.py` → local evaluation script
- `run_infer_qg_test_2ep.slurm` → Slurm QG job
- `slurm_infer_rg_2ep.sh` → Slurm RG job
- `slurm_infer_dg_2ep.sh` → Slurm DG job
- `slurm_score_all_2ep.sh` → Slurm full scoring job

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
NOTE: `retrieve.py` now has improvements from our team and is no longer the baseline retrieve.py
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
- Epochs trained: 1 and 2
- Final full-test inference used: `output/fullcoe_lora_v100_2ep`
- GPU: Tesla V100

Expected runtime:

```text
12–14 hours
```

## 5.8 Output
Training produced LoRA adapter checkpoints such as:

```text
output/fullcoe_lora_v100_1ep/
output/fullcoe_lora_v100_2ep/
```
These both contain:
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

# 5.9 Optional 1-Sample Sanity-Check Workflow

Before launching the full test-set pipeline, I also created a 1-sample QG → RG → DG workflow for debugging and validation.

This was useful for:
- confirming the finetuned model loaded correctly
- verifying the chained QG → RG → DG logic
- checking that evaluation worked on a tiny example before submitting long Slurm jobs

Scripts used for the 1-sample workflow:
- `infer_qg_test_2ep_1sample.py`
- `infer_rg_test_2ep_1sample.py`
- `infer_dg_test_2ep_1sample.py`
- `prepare_dg_1sample.py`
- `scoring_local_1sample.py`

Slurm scripts used for the 1-sample workflow:
- `slurm_infer_qg_1sample.sh`
- `slurm_infer_rg_1sample.sh`
- `slurm_infer_dg_1sample.sh`
- `slurm_score_1sample.sh`
- `slurm_score_dg_1sample.sh`

Important:
The 1-sample workflow was only used for debugging and sanity checking.
The final reported reproduction results come from the full test-set pipeline in Section 6.

---

# 6. Full CoE Inference Pipeline on the Full Test Set (QG → RG → DG)

After training, the Chain-of-Exemplar reproduction requires running the model in three chained stages on the ScienceQA test set:

QG → RG → DG

Important:
- This is the actual reproduction workflow used for evaluation.
- Long runs must be submitted through Slurm.
- The same finetuned LoRA checkpoint is reused across QG, RG, and DG.
- The later stages depend on files produced by earlier stages.

Checkpoint used for all full-test-set inference stages in this section:

`output/fullcoe_lora_v100_2ep`

This 2-epoch checkpoint was reused for:
- QG
- RG
- DG

---

## 6.1 Full-test-set files used

- Model checkpoint used for all stages:
  `output/fullcoe_lora_v100_2ep`
  
### QG
- Input:
  `data/ScienceQA_test_qg_blip2xl_angle.json`
- Inference script:
  `infer_qg_test_2ep.py`
- Slurm script:
  `run_infer_qg_test_2ep.slurm`
- Output predictions:
  `infer/pred_test_qg_2ep.json`

### RG
- Prep script:
  `prepare_rg_test_2ep.py`
- Generated RG test file:
  `data/ScienceQA_test_rg_from_qg_2ep.json`
- Inference script:
  `infer_rg_test_2ep.py`
- Slurm script:
  `slurm_infer_rg_2ep.sh`
- Output predictions:
  `infer/pred_test_rg_2ep.json`

### DG
- Prep script:
  `prepare_dg_test_2ep.py`
- Generated DG test file:
  `data/ScienceQA_test_dg_from_qg_rg_2ep.json`
- Inference script:
  `infer_dg_test_2ep.py`
- Slurm script:
  `slurm_infer_dg_2ep.sh`
- Output predictions:
  `infer/pred_test_dg_2ep.json`

### Evaluation
- Local scorer:
  `scoring_local.py`
- Full scoring Slurm script:
  `slurm_score_all_2ep.sh`

---

## 6.2 QG full-test inference

QG is the first stage.

Run:

```bash
sbatch run_infer_qg_test_2ep.slurm
```

Monitor:

```bash
squeue -u $USER
tail -n 40 logs/qg2ep_<JOBID>.err
```

Expected output file:

`infer/pred_test_qg_2ep.json`

Prediction format:

```json
[
  {"id": "qg_test_0", "response": "..."},
  ...
]
```

---

## 6.3 Build RG test set from QG outputs

RG is not run directly from the original test set.
Instead, the RG test set is built from the full QG predictions.

Run:

```bash
python prepare_rg_test_2ep.py
```

This creates:

`data/ScienceQA_test_rg_from_qg_2ep.json`

---

## 6.4 RG full-test inference

Run:

```bash
sbatch slurm_infer_rg_2ep.sh
```

Monitor:

```bash
squeue -u $USER
tail -n 40 logs/rg2ep_<JOBID>.err
```

Expected output file:

`infer/pred_test_rg_2ep.json`

RG predictions have the form:

```json
[
  {"id": "identity_test_0", "response": "..."},
  ...
]
```

---

## 6.5 Build DG test set from QG + RG outputs

DG is the final stage.
Its input depends on both the QG and RG outputs.

Run:

```bash
python prepare_dg_test_2ep.py
```

This creates:

`data/ScienceQA_test_dg_from_qg_rg_2ep.json`

---

## 6.6 DG full-test inference

Run:

```bash
sbatch slurm_infer_dg_2ep.sh
```

Monitor:

```bash
squeue -u $USER
tail -n 40 logs/dg2ep_<JOBID>.err
```

Expected output file:

`infer/pred_test_dg_2ep.json`

---

## 6.7 Important bug fix: malformed image paths in RG/DG datasets

While building the RG and DG test sets, some prompts contained malformed image paths like:

```text
<img>data/scienceqa/test/test_1/data/images/test/1.png</img>
```

These caused inference failures with `FileNotFoundError`.

They were patched so they instead use the correct path format:

```text
<img>data/images/test/1.png</img>
```

This patch was required for both:
- `data/ScienceQA_test_rg_from_qg_2ep.json`
- `data/ScienceQA_test_dg_from_qg_rg_2ep.json`

---

## 6.8 Slurm environment activation on WAVE

The correct activation pattern for these jobs is:

```bash
source /WAVE/apps/x86_64/packages/Anaconda3/2025.12-2/app/etc/profile.d/conda.sh
conda activate /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu
```

Typical Slurm directives used:

```bash
#SBATCH -J <jobname>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/<jobname>_%j.out
#SBATCH -e logs/<jobname>_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jloiretbernal@scu.edu
```

---

## 6.9 Full evaluation on the test set

Evaluation is run with the local scorer:

`scoring_local.py`

Metrics:
- BLEU-4
- METEOR
- ROUGE-L
- BLEURT

Because long scoring runs are slow on the cluster, final evaluation is submitted as a Slurm job:

```bash
sbatch slurm_score_all_2ep.sh
```

This scoring job runs:
1. QG scoring
2. RG scoring
3. DG scoring

in a single batch script.

---

## 6.10 Notes about scoring on HPC

BLEURT evaluation is the slowest part of scoring.

Important HPC notes:
- scoring may run on CPU even on a GPU node depending on TensorFlow/CUDA library availability
- NLTK data must be redirected outside the home directory due to quota limits
- Hugging Face and matplotlib caches should also be redirected to project storage

Environment variables used in scoring jobs:

```bash
export NLTK_DATA=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/nltk_data
export MPLCONFIGDIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/mplconfig
export HF_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache_scoreall
export TRANSFORMERS_CACHE=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache_scoreall
export HF_DATASETS_CACHE=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache_scoreall
export XDG_CACHE_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache_scoreall
```

# 7. RESULTS - BASELINE OF PAPER ON FULL TEST SET

This section summarizes the final full-test-set metrics for each stage of the Chain-of-Exemplar pipeline.

The final reporting table should contain three rows:
- QG
- RG
- DG

Each row reports:
- BLEU-4
- METEOR
- ROUGE-L
- BLEURT

| Stage | BLEU-4 | METEOR | ROUGE-L | BLEURT |
|------|--------|--------|---------|--------|
| QG   | 0.0431 | 0.2458 | 0.2505  | 0.3196 |
| RG   | 0.5572 | 0.6266 | 0.6381  | 0.6321 |
| DG   | 0.3931 | 0.6821 | 0.5891  | 0.5586 |

# 8. Final Full-Test Workflow Summary

The final full-test reproduction workflow is:

1. Build ScienceQA:
   `python build_scienceqa_problems.py`

2. Run CER:
   `python retrieve.py`

3. Build multitask train data:
   `python prepare_multitask.py`

4. Train LoRA:
   `sbatch slurm_fullcoe_lora.sbatch`

5. Run QG on full test set:
   `sbatch run_infer_qg_test_2ep.slurm`

6. Build RG test set:
   `python prepare_rg_test_2ep.py`

7. Run RG on full test set:
   `sbatch slurm_infer_rg_2ep.sh`

8. Build DG test set:
   `python prepare_dg_test_2ep.py`

9. Run DG on full test set:
   `sbatch slurm_infer_dg_2ep.sh`

10. Run final scoring:
   `sbatch slurm_score_all_2ep.sh`

# 9. Improvements

This section documents the CER improvements explored beyond the original baseline pipeline.
The current improvement experiments are implemented primarily in retrieve.py and were pushed to the josephine branch.

## 9.1 Improvement 1a: CER retrieval refinement

The first improvement focused on the Contextualized Exemplar Retrieval (CER) stage in `retrieve.py`.

Changes made:
- used answer-context similarity instead of answer-question similarity
- added modality-aware reranking
- added lightweight duplicate-question filtering
- preserved a fixed top-5 exemplar budget with fallback logic

This improvement was intended to reduce repetitive retrieved exemplars and improve contextual relevance.

## 9.2 Improvement 1b: CER question-aware retrieval

A second CER improvement was then tested in `retrieve.py`.

Additional changes made:
- added question-text similarity to the retrieval score
- kept answer similarity and context similarity
- kept modality-aware reranking
- kept lightweight diversity filtering for retrieved exemplars

This version was designed to improve semantic alignment between the input question and the retrieved exemplars.

## 9.3 Important rerun dependency

If `retrieve.py` is modified for an improvement experiment, the following steps must be rerun:

```bash
python retrieve.py
python prepare_multitask.py
sbatch slurm_fullcoe_lora.sbatch
sbatch run_infer_qg_test_2ep.slurm
python prepare_rg_test_2ep.py
sbatch slurm_infer_rg_2ep.sh
python prepare_dg_test_2ep.py
sbatch slurm_infer_dg_2ep.sh
sbatch slurm_score_all_2ep.sh
```

## 9.4 Improvement model outputs

Improvement 1a checkpoint (Can access on HPC):
  `/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar/output/fullcoe_lora_v100_2ep_retrievalfix`

Improvement 1b checkpoint:
  `add path after training completes`
