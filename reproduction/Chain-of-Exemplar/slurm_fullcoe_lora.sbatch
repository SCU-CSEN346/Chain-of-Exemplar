#!/bin/bash
#SBATCH --job-name=fullcoe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/fullcoe_%j.out
#SBATCH --mail-user=tkhambadkone@scu.edu
#SBATCH --mail-type=END,FAIL

bash finetune/run_fullcoe_lora_v100.sh
module purge
module load Anaconda3
module load CUDA/12.2.1

# Ensure conda is initialized and activate the correct environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate coe

# Diagnostics: print Python and deepspeed info
which python
python --version
python -c "import sys; print('Python executable:', sys.executable)"
python -c "import deepspeed; print('deepspeed location:', deepspeed.__file__)"

cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/Chain-of-Exemplar/reproduction/Chain-of-Exemplar

# Run the training script using the environment's Python
bash finetune/run_fullcoe_lora_v100.sh