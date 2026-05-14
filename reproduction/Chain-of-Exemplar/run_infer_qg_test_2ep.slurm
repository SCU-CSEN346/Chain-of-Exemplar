#!/bin/bash
#SBATCH --job-name=qg2ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/qg2ep_%j.out
#SBATCH --error=logs/qg2ep_%j.err
#SBATCH --mail-user=jloiretbernal@scu.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar

source ~/.bashrc
conda activate /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu

export HF_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export PIP_CACHE_DIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/pip_cache
export TMPDIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/tmp

mkdir -p logs infer

python infer_qg_test_2ep.py
