# README: Chain-of-Exemplar    
---
## Model Description: A short overview of the problem and your solution.
Option 2 - Improving upon Chain of Exemplar ([Luo et. al. 2024](https://aclanthology.org/2024.acl-long.432.pdf))

This project aims to address the lack of generated question diversity and mitigate hallucinations in the Chain of Exemplar model through in-context unsupervised learning, self-consistency, and potentially other methods that we come across throughout the quarter.

---
## Installation Instructions: List dependencies and how to install them.
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
This confirms:
- successful model loading
- GPU-based inference
- functional text generation pipeline

---
## Usage Instructions: How to run the main script and reproduce the results.
Load the dataset - [Dataset](https://huggingface.co/datasets/Lhh123/CoE_ScienceQA)

Dataset setup has been started using the authors’ released CoE ScienceQA dataset. Full loading and preprocessing instructions will be added after baseline dataset integration is finalized.

In progress - will fill in once we are able to get a baseline

---
## Expected Output: Example results or sample output.
In progress - will fill in once we are able to get a baseline

---
## Member Contributions: 

### JOSEPHINE - 33.3%: 
- [x] Set up project in HPC with VSCODE
- [x] Clone main branch to personal branch (Josephine)
- [x] Environment setup
- [x] Git setup including .gitignore (Commit + Push Git Setup)
- [x] Create Personal ReadMe with setup documentation + Add to Group ReadMe organization
- [x] Added GitHub + HF links to paper abstract and wrote In-Context Learning (2.2)
- [x] Got first baseline output by doing their quickstart with HF transformers and created instructions
- [x] Wrote Papers Methodology section in overleaf

### TARA - 33.3%: 
- [x] Set up project in HPC
- [x] Set up conda environment + documenting setup locally
- [x] Working on getting baseline to work on HPC
- [x] Model description in Group ReadMe
- [x] Wrote Automatic Multiple Choice Question Generation (2.1) and Self-Consistency (2.4) subsections in Related Works section in paper 

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
