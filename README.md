# Tara's README for Chain-of-Exemplar

## Table of Contents:
1. [Personal Contributions](#My-Contributions-Thus-Far)
2. [Repository Structure](#Repository-Structure)
3. [Quickstart Inference on HPC Guide](#How-to-Run-Quickstart-Inference-on-HPC)
4. [Full Self-Consistency Pipeline on HPC Guide](#How-to-Run-Full-Self-Consistency-Pipeline-on-HPC)

## My Contributions Thus Far
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
- [x] Ran Accuracy computation on baseline, CER 2, and self-consistency
- [x] Running final self-consistency experiments with new hyperparameter configurations
- [x] Contributed to Limitations section in slides and paper
- [x] Edited Related Works section in paper according to earlier feedback
- [x] Created self-consistency diagram
- [x] Contributed to the poster presentation and powerpoint presentation
- [x] Created interactive demo

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
- `infer_rg_test_2ep.py` → full-test RG inference (regular RG)
- `infer_self_consistency.py` → self-consistency RG inference
- `infer_self_consistency_parallel.py` → 2 GPU parallelized self-consistency RG inference
- `infer_self_consistency_batch.py` → batch self-consistency RG inference
- `prepare_dg_test_2ep.py` → build DG full-test input from QG + RG outputs
- `prepare_dg_test_2ep_self_consistency.py` → build DG full-test input from QG + Self-Consistent RG outputs
- `infer_dg_test_2ep.py` → full-test DG inference
- `scoring_local.py` → local LLM evaluation script
- `extract_distractors.py` → distractors extraction script for Accuracy scoring script
- `scoring_accuracy.py` → Accuracy scoring script
- `run_infer_qg_test_2ep.slurm` → Slurm QG job
- `slurm_infer_rg_2ep.sh` → Slurm RG job (regular RG)
- `run_infer_self_consistency.slurm` → Slurm self-consistency RG job
- `run_infer_self_consistency_parallel.slurm` → Slurm 2 GPU parallelized self-consistency RG job
- `run_infer_self_consistency_batch.slurm` → Slurm batch self-consistency RG job
- `slurm_infer_dg_2ep.sh` → Slurm DG job
- `slurm_score_all_2ep.sh` → Slurm LLM metrics scoring job
- `run_scoring_accuracy.sh` → Slurm Accuracy scoring job

---

## How to Run Quickstart Inference on HPC

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
  deepspeed==0.9.5 \
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

## How to Run Full Self-Consistency Pipeline on HPC
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

### 1. Full-test-set files used

- Model checkpoint used for all stages:
  `output/fullcoe_lora_v100_2ep`
  
#### QG
- Input:
  `data/ScienceQA_test_qg_blip2xl_angle.json`
- Inference script:
  `infer_qg_test_2ep.py`
- Slurm script:
  `run_infer_qg_test_2ep.slurm`
- Output predictions:
  `infer/pred_test_qg_2ep.json`

#### Self-Consistent RG
- Prep script:
  `prepare_rg_test_2ep.py`
- Generated RG test file:
  `data/ScienceQA_test_rg_from_qg_2ep.json`
- Inference script:
  `infer_self_consistency.py` or `infer_self_consistency_parallel.py` or  `infer_self_consistency_batch.py`
- Slurm script:
  `run_infer_self_consistency.slurm` or `run_infer_self_consistency_parallel.slurm` or `run_infer_self_consistency_batch.slurm`
- Output predictions:
  `infer/rationales_self_consistent.json`

#### DG
- Prep script:
  `prepare_dg_test_2ep_self_consistency.py`
- Generated DG test file:
  `data/ScienceQA_test_dg_from_qg_rg_2ep.json`
- Inference script:
  `infer_dg_test_2ep.py`
- Slurm script:
  `slurm_infer_dg_2ep.sh`
- Output predictions:
  `infer/pred_test_dg_2ep.json`

#### Evaluation
- Local NLP metrics scorer:
  `scoring_local.py`
- Full NLP metrics scoring Slurm script:
  `slurm_score_all_2ep.sh`
- Distractors extraction for Accuracy scorer:
  `extract_distractors.py`
- Accuracy scorer:
  `scoring_accuracy.py`
- Accuracy scoring Slurm script:
  `run_scoring_accuracy.sh`

---

### 2. QG full-test inference

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

### 3. Build RG test set from QG outputs

RG is not run directly from the original test set.
Instead, the RG test set is built from the full QG predictions.

Run:

```bash
python prepare_rg_test_2ep.py
```

This creates:

`data/ScienceQA_test_rg_from_qg_2ep.json`

---

### 4. Self-Consistent RG full-test inference

Run:

```bash
sbatch run_infer_self_consistency.slurm
```

Monitor:

```bash
squeue -u $USER
tail -n 40 logs/self_consistency_rg_<JOBID>.err
```

Expected output file:

`infer/rationales_self_consistent.json`

Self-Consistent RG predictions have the form:

```json
[
  {"id": "identity_test_0",
   "question": "...",
   "best_rationale": "...",
   "all rationales": ["...", "...", "..."],
   "scores": [float, float, float]
  },
]
```

---

### 5. Build DG test set from QG + Self-Consistent RG outputs

DG is the final stage.
Its input depends on both the QG and RG outputs.

Run:

```bash
python prepare_dg_test_2ep_self_consistency.py
```

This creates:

`data/ScienceQA_test_dg_from_qg_rg_2ep.json`

---

### 6. DG full-test inference

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
### 7. Full evaluation on the test set

#### LLM Evaluation Metrics
LLM evaluation is run with the local scorer:

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

#### Accuracy Score
We need to extract distractors from the test set before running Accuracy scoring:

```bash
python extract_distractors.py   --pred infer/pred_test_dg_2ep.json   --gt data/ScienceQA_test_dg_from_qg_rg_2ep.json   --output extracted_distractors.json
```

This outputs the file `extracted_distractors.json`

Accuracy scoring is run using the script:
`scoring_accuracy.py`

The Accuracy scoring script must be run using a SLURM script:
```bash
sbatch run_scoring_accuracy.sh
```
---
## Full Fine-Tuning + Self-Consistency Workflow Summary

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

7. Run Self-Consistent RG on full test set:
   `sbatch run_infer_self_consistency.slurm`

8. Build DG test set:
   `python prepare_dg_test_2ep_self_consistency.py`

9. Run DG on full test set:
   `sbatch slurm_infer_dg_2ep.sh`

10. Run LLM metrics scoring:
   `sbatch slurm_score_all_2ep.sh`

11. Extract distractors for Accuracy scoring:
    `python extract_distractors.py`

12. Run Accuracy scoring:
    `sbatch run_scoring_accuracy.sh`

