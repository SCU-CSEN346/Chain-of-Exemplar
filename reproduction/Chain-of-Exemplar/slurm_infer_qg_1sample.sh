#!/bin/bash
#SBATCH -J qg1sample
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/qg1sample_%j.out
#SBATCH -e logs/qg1sample_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jloiretbernal@scu.edu

mkdir -p logs
cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar

source /WAVE/apps/x86_64/packages/Anaconda3/2025.12-2/app/etc/profile.d/conda.sh
conda activate /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu

export NLTK_DATA=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/nltk_data

python infer_qg_test_2ep_1sample.py
