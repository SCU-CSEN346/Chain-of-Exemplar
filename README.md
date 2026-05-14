# Chain-of-Exemplar Reproduction

⚠️ Note: The original repository is not directly reproducible. See below for all fixes applied.

## :rocket: Overview
This repo contains our attempt to reproduce the CoE paper. Due to multiple issues in the original repository, significant effort was required to make the environment reproducible and run the baseline.

## :gear: Environment Setup
- OS: WSL (Ubuntu)
- Python: 3.10
- GPU: NVIDIA RTX 3090 (CUDA enabled)
- Framework: PyTorch

## ❗ Issues with Original Repository
The provided <kbd>requirements.txt</kbd> was not directly usable due to:
### 1. Missing / Invalid Packages
These packages were not available on PyPI:

- bliva==1.0.0
- clip==1.0
- clip-flant5==1.1.2
- distinct-n==0.4.0

 ➡️ Solution: Commented out

### 2. Hardware-Specific / Optional Dependencies
These required complex builds or were not needed for baseline:

- detectron2==0.6
- flash-attn==2.6.3
- salesforce-lavis==1.0.1

 ➡️ Solution: Commented out

### 3. Invalid / Dev Versions
evaluate==0.4.1.dev0 → replaced with evaluate==0.4.1

### 4. Python Compatibility Issues
typing==3.10.0.0 → removed (built-in)
backports.zoneinfo==0.2.1 → removed (not needed for Python ≥3.9)

### 5. Dependency Conflicts
| Package      | Issue                   | Fix                 |
| ------------ | ----------------------- | ------------------- |
| transformers | conflict with angle-emb | updated to 4.32.1   |
| accelerate   | conflict with auto-gptq | updated to 0.22.0   |
| peft         | outdated version        | updated to 0.5.0    |
| typer        | incompatible with spacy | downgraded to 0.9.0 |

### 6. Core Conflict (NumPy + TensorFlow)
<kbd>numpy==1.24.4</kbd> conflicted with TensorFlow
TensorFlow caused further conflicts with <kbd>typing-extensions</kbd>

➡️ Solution:
- downgraded numpy → <kbd>1.24.3</kbd>
- removed TensorFlow (not required)

## 🤖 Model Issue
We attempted to run the baseline inference using the original CoE repository. However, the required base model (Qwen-VL-Chat-Int4) was unavailable. To proceed, we replaced it with a publicly available model (Qwen-1.8B-Chat) to validate the inference pipeline.

The original model depends on:
```text
Qwen-VL-Chat-Int4
```
This model:

- is not publicly accessible
- results in 404 errors

## ✔️ Baseline Execution
To proceed, we replaced the base model with:
```text
Qwen/Qwen-1_8B-Chat
```
### Sample Inference
#### Input:
```text
Answer: gravity  
Please generate a question from the corresponding answer.
```
#### Output:
```text
What is the force of gravity that attracts two objects with mass towards each other?
```

🎯 Key Takeaways
- The original repository is not directly reproducible
- Multiple dependencies are outdated or unavailable
- Model availability is a major issue
- Despite this, we successfully:
  - set up environment
  - resolved conflicts
  - ran baseline inference

🚀 Next Steps
- Just ran a text input for now, next step is to run a multimodal input
- Implement full CoE pipeline
- Integrate ScienceQA dataset
- Improve:
 - reasoning (reduce hallucination)
 - diversity (better distractors)
 - self-consistency

-----------

## 🔬 Local Full CoE Reproduction Progress (RTX 3090)

### ✔️ Completed Steps

#### 1. Local Environment Setup
Created isolated Python environment:
```bash
python3 -m venv coe_local
source coe_local/bin/activate
```

Verified GPU access and name:
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Output:
```text
True
NVIDIA GeForce RTX 3090
```

Installed compatible NumPy version:
```bash
pip install "numpy<2"
```

### ✔️ ScienceQA Dataset Build
Generated reversed ScienceQA dataset locally:

```bash
python build_scienceqa_problems.py
```

Generated:
```text
data/scienceqa/problems.json
```

Dataset size:
```text
21208 samples
```

### ✔️ Contextualized Exemplar Retrieval (CER)

#### Original Issue
The original retrieval pipeline had multiple issues:
- missing angle_emb
- deprecated HuggingFace APIs
- invalid retrieval thresholds
- retrieval runtime was extremely slow (~11 hrs)
- retrieval produces 0 samples

#### Fixes Applied
Installed:
```bash
pip install angle-emb
```
Fixed:
- deprecated 'huggingface_hub' usage
- incorrect 'data_root'
- retrieval thresholding
- missing embedidng generation
- invalid retrieval save paths

#### Retrieval Runtime Optimization

Original Implementation:
- pairwise Python similarity loops
- runtime estimated 7-11 hrs

Improved Implemantation:
- vectorized embedding similarity computation
- GPU-based retrieval scoring

Runtime after optimization:
```text
~ 3 seconds
```

#### Retrieval Verification

Generated:

```text
data/scienceqa/problems_blip2xl_angle.json
```

Verified retrieved exemplars:

```text
with relevant_question: 4373
```

Example:

```text
train_3:
['train_637', 'train_850', 'train_1058', 'train_5039', 'train_5447']
```

### ✔️ QG Dataset Preparation

#### Original Issues:
'prepare_qg.py' contained:
- incorrect split names ('val' instead of 'validation')
- broken multimodal image paths
- empty generated datasets

#### Fixes Applied
Changed;
```python
split = 'val
```
to:
```pythin
split = 'validation'
```

Fixed image formatting from:
```python
os.path.join(data_root, ...)
```
to direct image paths:
```python
problem['image']
```

#### Generated Validation Dataset
Command:
```bash
python prepare_qg.py
```

Generated:
```text
data/ScienceQA_validation_qg_norationale.json
```
Samples = 4241

#### Generated Training Dataset
Generated:
```text
data/ScienceQA_train_qg_norationale.json
```
Samples = 12726

Example Prompt:
```text
Picture: <img>data/images/train/0.png</img>
Answer: West Virginia
generate a question based on the above picture and the corresponding answer.
```

### ✔️ LoRA Fine-Tuning Setup

Created local QG LoRA training script:
```bash
cp finetune/finetune_lora_single_gpu.sh finetune/train_qg_lora_local.sh
```

Updated:
- dataset paths
- output paths
- epochs
- local output directories

Installed DeepSpeed:
```bash
pip install deepspeed==0.10.3
```
Verified Qwen tokenizer loading:
```bash
python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    trust_remote_code=True
)
print("tokenizer loaded")
PY
```

### ✔️ Tiny QG LoRA Sanity Training

Created tiny dataset:

```bash
data/ScienceQA_train_qg_tiny20.json
```

Tiny training successfully completed using:
- Qwen-VL-Chat
- LoRA
- RTX 3090

Training logs:

```text
{'loss': 1.6989, 'learning_rate': 1e-05, 'epoch': 0.4}
{'loss': 1.6007, 'learning_rate': 0.0, 'epoch': 0.8}
```

This verified:
- multimodal dataset formatting
- image loading
- Qwen-VL loading
- LoRA compatibility
- local GPU training pipeline

## 🎯 DG (Distractor Generation) Improvement

### Motivation

Baseline CoE distractor generation often:
- generated only one distractor
- produced semantically weak distractors
- generated repetitive or trivial distractors

### Proposed DG Prompt Improvements

Modified prompts to encourage:
- multiple distractors
- grammatical consistency
- semantic similarity to the correct answer
- reduced trivial distractors
- improved distractor plausibility

### Current Status

Prompt reformulations implemented locally.
Full end-to-end DG evaluation using generated QG/RG outputs is currently running.

### ✔️ Full QG LoRA Fine-Tuning

Successfully completed local QG LoRA fine-tuning using:
```bash
bash finetune/train_qg_lora_local.sh
```
Configuration:
- Qwen/Qwen-VL-Chat
- LoRA fine-tuning
- 1 epoch
- RTX 3090

Runtime:
```text
~12 hrs
```

Generated output:
```text
output/qg_lora_local
```

Saved artifacts:
- adapter_model.bin
- adapter_config.json
- checkpoint-1000
- tokenizer files
- trainer state

### ✔️ QG Inference Verification

Created local QG inference pipeline:

```text
infer_qg_local.py
```

Verified:
- model loading
- multimodal image loading
- LoRA adapter inference
- validation dataset formatting
- generated predictions

Tiny validation inference test:

```text
5/5 samples completed successfully
```

Example prediction:

```text
Input Answer:
"The snoring is loud."

Generated Question:
"What information supports the conclusion that the snoring is loud?"
```

### ✔️ Full Validation of QG Inference

Completed:
```bash
python infer_qg_local.py
```

Generated Output:
```text
infer/pred_validation_qg_local.json
```

Runtime = ~50 minutes

Samples generated = 4241

Example prediction:
```text
Input Answer:
"The snoring is loud."

Generated Question:
"What information supports the conclusion that John is a snorer?"
```

The generated outputs are now used for downstream RG dataset preparation and evaluation.


### ✔️ RG Dataset Preparation

Successfully generated local RG validation dataset using generated QG outputs.
Pipeline:
```text
QG predictions -> RG dataset construction
```

Generated file:
```text
data/ScienceQA_validation_rg_locak.json
```

Samples generated = 4241

The dataset is used for downstream rationale generation (RG) fine-tuning and inference.

## ⏳ Currently Running

### Full Train-Set QG Inference

Running:
```bash
python infer_qg_train_local.py
```

Output target:
```text
infer/pred_train_qg_local.json
```
Estimated Runtime: ~2-3 hrs

## ⏭️ Remaining Steps

- [x] Complete full validation QG inference
- [x] Prepare RG validation datasets using generated QG outputs
- [ ] Run full train-set QG inference
- [ ] Prepare RG training dataset
- [ ] Fine-tune RG model
- [ ] Run RG inference
- [ ] Prepare DG baseline datasets
- [ ] Implement improved DG prompt variants
- [ ] Run DG baseline inference
- [ ] Run DG improved inference
- [ ] Evaluate BLEU-4 / METEOR / ROUGE-L / BLEURT
- [ ] Compare baseline vs improved DG outputs
- [ ] Finalize qualitative distractor comparisons

## ⚠️ Semantic Drift Observation

During local reproduction, we observed that generated questions occasionally preserved compatibility with the correct answer while changing the underlying educational objective of the original question.

Example:
- original question tested verbal irony
- generated question shifted toward factual inference
- answer remained technically correct

This suggests that standard lexical evaluation metrics alone may not fully capture pedagogical fidelity in educational MCQ generation pipelines.

# ⚠️ Notes
Due to compute and runtime limitations, some experiments are still in progress. Current work focuses on achieving a fully reproducible local CoE pipeline and improving distractor generation quality.
