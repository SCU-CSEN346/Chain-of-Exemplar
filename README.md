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


