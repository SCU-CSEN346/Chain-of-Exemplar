#!/bin/bash
#SBATCH --job-name=self_consistency_rg
#SBATCH --output=logs/self_consistency_rg_%j.out
#SBATCH --error=logs/self_consistency_rg_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --mail-user=tkhambadkone@scu.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

module purge
module load Anaconda3
module load CUDA/12.2.1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate coe

# Set paths (edit these as needed)
MODEL_PATH="output/fullcoe_lora_v100_2ep_retrievalfix"
DATA_PATH="data/ScienceQA_test_rg_from_qg_2ep.json"  # Input questions
OUTPUT_PATH="infer/rationales_self_consistent.json"  # Output file
NUM_SAMPLES=3

# Run self-consistency inference
/WAVE/users2/unix/tkhambadkone/.conda/envs/coe/bin/python infer_self_consistency.py \
  --model_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --output_path "$OUTPUT_PATH" \
  --num_return_sequences $NUM_SAMPLES \
  --max_new_tokens 128 \
  --min_new_tokens 8 \
  --top_p 0.95 \
  --top_k 20 \
  --temperature 0.7 \
  --repetition_penalty 1.4 \
  --no_repeat_ngram_size 4 