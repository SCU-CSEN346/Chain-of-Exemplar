# Josephine's ReadMe for Chain-of-Exemplar
## WHAT JOSEPHINE HAS DONE THUS FAR: 
- [x] Set up project in HPC with VSCODE
- [x] Clone main branch to personal branch (Josephine)
- [x] Environment setup
- [x] Git setup (Commit + Push Git Setup)
- [x] Create Personal ReadMe + Add to Group ReadM eorganization
- [x] Added GitHub + HF links to paper abstract and wrote In-Context Learning (2.2)
      
---
## TO BE COMPLETED THROUGHOUT SUBMISSIONS: 
### Model Description: Not yet applicable
### Installation Instructions: Not yet applicable
### Usage Instructions: Below
### Expected Output: Not yet applicable
### Member Contributions: Not yet applicable

---

## HPC Environment Setup

### Step 1. Connect to HPC and request GPU

```bash
ssh your_username@login01
srun -p gpu --gres=gpu:1 --time=02:00:00 --pty bash
```

Verify GPU:

```bash
nvidia-smi
```

### Step 2. Load Anaconda

```bash
module purge
module load Anaconda3
```

### Step 3. Create environment in project storage (IMPORTANT TO NOT EXCEED STORAGE QUOTA)

```bash
conda create --prefix /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu python=3.10 -y
conda activate /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu
```

### Step 4. Install PyTorch (GPU version)

Use **pip** (avoids HPC library conflicts):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Expected output:** `True`

### Step 5. Install required libraries

Only install these and not entire requirements: 

```bash
pip install transformers datasets tqdm peft
```

---

## Git Setup for HPC

### Step 1. Create `.gitignore`

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

### Step 2. Commit `.gitignore`

```bash
git add .gitignore
git commit -m "add gitignore to ignore venv"
git push
```
