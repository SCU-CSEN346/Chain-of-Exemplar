#!/bin/bash
#SBATCH -J rg2ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/rg2ep_%j.out
#SBATCH -e logs/rg2ep_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tkhambadkone@scu.edu

set -euo pipefail

module purge
module load Anaconda3
module load CUDA/12.2.1

cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/Chain-of-Exemplar/reproduction/Chain-of-Exemplar

source $(conda info --base)/etc/profile.d/conda.sh
conda activate coe

/WAVE/users2/unix/tkhambadkone/.conda/envs/coe/bin/python infer_rg_test_2ep.py
