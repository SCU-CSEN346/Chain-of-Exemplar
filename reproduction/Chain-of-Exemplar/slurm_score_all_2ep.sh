#!/bin/bash
#SBATCH -J scoreall2ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH -o logs/scoreall2ep_%j.out
#SBATCH -e logs/scoreall2ep_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jloiretbernal@scu.edu

set -euo pipefail

source /WAVE/apps/x86_64/packages/Anaconda3/2025.12-2/app/etc/profile.d/conda.sh
conda activate /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu

cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar

mkdir -p /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/nltk_data
mkdir -p /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/mplconfig
mkdir -p /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache_scoreall

export NLTK_DATA=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/nltk_data
export MPLCONFIGDIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/mplconfig
export HF_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache_scoreall
export TRANSFORMERS_CACHE=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache_scoreall
export HF_DATASETS_CACHE=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache_scoreall
export XDG_CACHE_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache_scoreall

echo "===== QG ====="
python scoring_local.py \
  --gt data/ScienceQA_test_qg_blip2xl_angle.json \
  --pred infer/pred_test_qg_2ep.json \
  --problems data/scienceqa/problems_blip2xl_angle.json

echo "===== RG ====="
python scoring_local.py \
  --gt data/ScienceQA_test_rg_from_qg_2ep.json \
  --pred infer/pred_test_rg_2ep.json \
  --problems data/scienceqa/problems_blip2xl_angle.json

echo "===== DG ====="
python scoring_local.py \
  --gt data/ScienceQA_test_dg_from_qg_rg_2ep.json \
  --pred infer/pred_test_dg_2ep.json \
  --problems data/scienceqa/problems_blip2xl_angle.json
