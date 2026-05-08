#!/bin/bash
#SBATCH --job-name=fullcoe_2ep_retrievalfix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/fullcoe_2ep_retrievalfix_%j.out
#SBATCH --mail-user=jloiretbernal@scu.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar

source ~/.bashrc
conda activate /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu

export HF_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

bash finetune/run_fullcoe_lora_v100.sh
