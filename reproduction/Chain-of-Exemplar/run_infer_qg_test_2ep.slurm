#!/bin/bash
#SBATCH --job-name=qg2ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/qg2ep_%j.out
#SBATCH --error=logs/qg2ep_%j.err
#SBATCH --mail-user=tkhambadkone@scu.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

module purge
module load Anaconda3
module load CUDA/12.2.1

# Ensure conda is initialized and activate the correct environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate coe

cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/Chain-of-Exemplar/reproduction/Chain-of-Exemplar

# export HF_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/hf_cache
# export TRANSFORMERS_CACHE=$HF_HOME
# export HF_DATASETS_CACHE=$HF_HOME
# export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
# export PIP_CACHE_DIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/pip_cache
# export TMPDIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/tmp

mkdir -p logs infer

/WAVE/users2/unix/tkhambadkone/.conda/envs/coe/bin/python infer_qg_test_2ep.py
