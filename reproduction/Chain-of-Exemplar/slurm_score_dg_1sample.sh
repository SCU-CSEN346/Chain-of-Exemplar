#!/bin/bash
#SBATCH -J dg1score
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/dg1score_%j.out
#SBATCH -e logs/dg1score_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jloiretbernal@scu.edu

mkdir -p logs
cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar

source /WAVE/apps/x86_64/packages/Anaconda3/2025.12-2/app/etc/profile.d/conda.sh
conda activate /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu

export NLTK_DATA=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/nltk_data

python scoring_dg_1sample.py \
  --gt data/ScienceQA_test_dg_from_qg_rg_2ep_1sample.json \
  --pred infer/pred_test_dg_2ep_1sample.json
