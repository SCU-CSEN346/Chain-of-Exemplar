#!/bin/bash
#SBATCH --job-name=self_consistency
#SBATCH --array=0-1          # 2 jobs, one per GPU
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1         # 1 GPU per job
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
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
OUTPUT_PATH="infer/rationales_self_consistent_parallel.json"  # Output file
NUM_SAMPLES=5

/WAVE/users2/unix/tkhambadkone/.conda/envs/coe/bin/python infer_self_consistency_parallel.py \
  --model_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --output_path "$OUTPUT_PATH" \
  --num_return_sequences $NUM_SAMPLES \
  --max_new_tokens 500 \
  --min_new_tokens 8 \
  --top_p 0.95 \
  --top_k 20 \
  --temperature 0.7 \
  --repetition_penalty 1.4 \
  --no_repeat_ngram_size 4 \
  --num_return_sequences $NUM_SAMPLES \
  --shard_id ${SLURM_ARRAY_TASK_ID} \
  --num_shards 2 \

# #!/bin/bash
# #SBATCH --job-name=self_consistency_parallel
# #SBATCH --output=logs/self_consistency_parallel_%j.out
# #SBATCH --error=logs/self_consistency_parallel_%j.err
# #SBATCH --nodes=1
# #SBATCH --ntasks=2
# #SBATCH --cpus-per-task=6
# #SBATCH --gres=gpu:2
# #SBATCH --mem=64G
# #SBATCH --time=24:00:00
# #SBATCH --partition=gpu
# #SBATCH --mail-user=tkhambadkone@scu.edu
# #SBATCH --mail-type=END,FAIL,TIME_LIMIT

# module purge
# module load Anaconda3
# module load CUDA/12.2.1

# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate coe

# # Split input (edit paths as needed)
# python split_json.py data/ScienceQA_test_rg_from_qg_2ep.json input_split 2

# # Run inference in parallel on 2 GPUs
# srun --exclusive -N1 -n1 bash -c "CUDA_VISIBLE_DEVICES=0 python infer_self_consistency.py --model_path output/fullcoe_lora_v100_2ep_retrievalfix --data_path input_split_0.json --output_path output_0.json --num_samples 5 --max_new_tokens 256 --top_p 0.95 --temperature 0.7" &
# srun --exclusive -N1 -n1 bash -c "CUDA_VISIBLE_DEVICES=1 python infer_self_consistency.py --model_path output/fullcoe_lora_v100_2ep_retrievalfix --data_path input_split_1.json --output_path output_1.json --num_samples 5 --max_new_tokens 256 --top_p 0.95 --temperature 0.7" &
# wait

# # Merge outputs
# python merge_json.py merged_output.json output_0.json output_1.json

# echo "Self-consistency inference complete. Results in merged_output.json."
