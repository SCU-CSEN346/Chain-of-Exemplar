# Chain-of-Exemplar — Group README

This README provides the setup and execution instructions needed to reproduce our Chain-of-Exemplar pipeline. 

## Table of Contents

- [Model Description](#model-description-chain-of-exemplar-for-mcq-generation)
- [Member Contributions](#member-contributions)
- [Installation Instructions](#installation-instructions)
- [Usage Instructions](#usage-instructions)
- [Expected Output](#expected-output)
- [Improvements](#improvements)

---

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
- [ ] Finalize improvements
- [ ] Finalize Github 
- [ ] Finalize Paper (Get under 6 pages!) 
- [x] Create Demo (maybe use the 1 sample scripts Josephine created - remember to change model to improved one)
- [ ] Create Presentation 
- [ ] Create Poster


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
- [x] Improve Readme for personal branch and main branch
- [x] Merge my branch with main
- [x] Paper: Contributed to the following sections: Final Results, Limitations, Conclusion
- [x] Filmed and edited the demo

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
- [x] Wrote self-consistency rationale generation scripts (1 GPU, 2 GPUs, and batched/chunked)
- [x] Extensive prompt engineering and hyperparameter tuning to make Qwen tokenizer compatible with self-consistency and reduce hallucinations at generation time
- [x] Described self-consistency in paper (7.2)
- [x] Began ethics discussion in paper (10)
- [x] Implemented Accuracy metric scripts

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
- [x] Reproduced Contextualized Exemplar Retrieval (CER) locally and generated relevant_question mappings for ScienceQA
- [x] Reduced retrieval runtime from several hours to seconds through retrieval debugging and fixes
- [x] Built local QG inference pipeline and generated full validation-set QG outputs
- [x] Generated RG validation dataset using generated QG outputs
- [x] Observed semantic drift where generated questions preserved answers but changed the educational objective
- [x] Implementing DG prompt improvements for more plausible and semantically consistent distractors
- [x] Completed section 7.3 (Improvement 3: Enhanced Distractor Generation) in the overleaf document
- [x] Added to the Limitations and Ethics section of the document

---

# Installation Instructions

These instructions assume a Linux/HPC environment with access to an NVIDIA GPU. Our full reproduction was run on SCU WAVE using Python 3.10, CUDA-compatible PyTorch, Qwen/Qwen-VL-Chat, LoRA fine-tuning, and Slurm jobs.

## 1. Clone the repository

```bash
git clone https://github.com/SCU-CSEN346/Chain-of-Exemplar.git
cd Chain-of-Exemplar/reproduction/Chain-of-Exemplar
```

## 2. Create and activate the conda environment

On WAVE, load Anaconda first:

```bash
module purge
module load Anaconda3
```

Create and activate the environment. On shared HPC storage, prefer a project path rather than home storage.

```bash
conda create --prefix /path/to/coe_gpu python=3.10 -y
conda activate /path/to/coe_gpu
```

## 3. Set cache paths

Set cache directories outside the home directory to avoid quota errors. Replace the paths below with your own project storage.

```bash
export HF_HOME=/path/to/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export SENTENCE_TRANSFORMERS_HOME=$HF_HOME/sentence_transformers
export PIP_CACHE_DIR=/path/to/pip_cache
export TMPDIR=/path/to/tmp
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$SENTENCE_TRANSFORMERS_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
```

## 4. Install dependencies

Install PyTorch with CUDA 11.8 support:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

Install the required packages:

```bash
pip install transformers==4.32.0 peft==0.4.0 accelerate==0.21.0 datasets tqdm sentencepiece einops matplotlib tiktoken transformers_stream_generator "numpy<2"
pip install deepspeed==0.9.5
pip install pillow jsonlines requests==2.32.3
pip install huggingface-hub==0.25.2
pip install sentence-transformers==2.2.2
```

Important notes:
- Use `Qwen/Qwen-VL-Chat` as the base model.
- Do **not** use `Qwen/Qwen-VL-Chat-Int4` for this setup.
- Do **not** install the newest `sentence-transformers`, because it may upgrade `transformers` and break the Qwen-VL stack.

## 5. Verify versions

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

Expected versions:

```text
torch 2.0.1+cu118
transformers 4.32.0
huggingface-hub 0.25.2
sentence-transformers 2.2.2
```

---

# Usage Instructions

All commands should be run from:

```bash
cd /path/to/Chain-of-Exemplar/reproduction/Chain-of-Exemplar
conda activate /path/to/coe_gpu
```

For GPU jobs on WAVE or another cluster, use Slurm rather than running long jobs on the login node.

## Full CoE baseline pipeline

Run the full pipeline in this order:

| Step | Command | Purpose |
|------|---------|---------|
| 1 | `python build_scienceqa_problems.py` | Build ScienceQA problem files |
| 2 | `python retrieve.py` | Run Contextualized Exemplar Retrieval (CER) |
| 3 | `python prepare_multitask.py` | Build multitask QG/RG/DG training data |
| 4 | `sbatch slurm_fullcoe_lora.sbatch` | Fine-tune the LoRA model |
| 5 | `sbatch run_infer_qg_test_2ep.slurm` | Run question generation on the full test set |
| 6 | `python prepare_rg_test_2ep.py` | Build RG test inputs from QG outputs |
| 7 | `sbatch slurm_infer_rg_2ep.sh` | Run rationale generation |
| 8 | `python prepare_dg_test_2ep.py` | Build DG test inputs from QG + RG outputs |
| 9 | `sbatch slurm_infer_dg_2ep.sh` | Run distractor generation |
| 10 | `sbatch slurm_score_all_2ep.sh` | Score QG, RG, and DG outputs |

Monitor jobs with:

```bash
squeue -u $USER
tail -n 40 logs/<jobname>_<JOBID>.err
```

## Important path note for RG/DG

If RG or DG fails with a `FileNotFoundError` involving paths like:

```text
data/scienceqa/test/test_1/data/images/test/1.png
```

patch the generated RG/DG JSON so image paths use the shorter format:

```text
data/images/test/1.png
```

The same issue can occur for `train`, `validation`, and `test` image paths. See Josephine's personal README for the detailed debugging notes and exact patch commands.

---

# Expected Output

## Main generated files

| Stage | Output file |
|-------|-------------|
| QG | `infer/pred_test_qg_2ep.json` |
| RG | `infer/pred_test_rg_2ep.json` |
| DG | `infer/pred_test_dg_2ep.json` |
| Scoring | `logs/scoreall2ep_<JOBID>.out` |

The prediction files are JSON lists with entries similar to:

```json
{"id": "qg_test_0", "response": "What is the apostrophe used for?"}
```

## Baseline full-test metrics

The 2-epoch LoRA baseline reproduced on the full ScienceQA test set produced:

| Stage | BLEU-4 | METEOR | ROUGE-L | BLEURT |
|------|--------|--------|---------|--------|
| QG | 0.0431 | 0.2458 | 0.2505 | 0.3196 |
| RG | 0.5572 | 0.6266 | 0.6381 | 0.6321 |
| DG | 0.3931 | 0.6821 | 0.5891 | 0.5586 |

---

# Improvements

This section briefly summarizes improvement experiments beyond the baseline. More detailed notes, debugging history, and exact implementation details are available in the relevant team member branches and personal READMEs.

## Josephine: CER retrieval improvements

Josephine's improvement focused on the Contextualized Exemplar Retrieval (CER) stage in `retrieve.py`. The final version added question-aware retrieval scoring while keeping answer/context similarity, modality-aware reranking, and lightweight diversity filtering.

The best CER improvement run improved RG and DG compared with the baseline, while QG showed mixed results.

| Stage | BLEU-4 | METEOR | ROUGE-L | BLEURT |
|------|--------|--------|---------|--------|
| QG | 0.0355 | 0.2512 | 0.2524 | 0.3205 |
| RG | 0.6896 | 0.7375 | 0.7556 | 0.6908 |
| DG | 0.4313 | 0.7384 | 0.6156 | 0.5886 |

Shared Improvement 1b checkpoint on WAVE:

```text
/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar/output/fullcoe_lora_v100_2ep_retrievalfix
```

## Tara

_To be added by Tara._

## Kajal

_To be added by Kajal._
