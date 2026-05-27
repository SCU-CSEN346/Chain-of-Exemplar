#!/bin/bash
#SBATCH --job-name=coe_scoring_accuracy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/scoring_accuracy_%j.out
#SBATCH --error=logs/scoring_accuracy_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jloiretbernal@scu.edu

# ─────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────

PROJECT_DIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar
QA_MODEL=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_multitask_blip2xl_angle_2ep
PROBLEMS=$PROJECT_DIR/problems_fixed.json
IMAGE_DIR=$PROJECT_DIR/data/images
EXTRACTED=$PROJECT_DIR/extracted_distractors.json
OUTPUT=$PROJECT_DIR/accuracy_results.json
SCRIPT=$PROJECT_DIR/scoring_accuracy.py

CONDA_ENV=coe_gpu

# ─────────────────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────────────────

echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "GPU(s):       $CUDA_VISIBLE_DEVICES"
echo "Start time:   $(date)"
echo "=========================================="

# Load conda and activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

# Confirm GPU is visible
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Verify required files exist before starting
echo "Checking required files..."
for f in "$EXTRACTED" "$PROBLEMS" "$SCRIPT" "$QA_MODEL"; do
    if [ ! -e "$f" ]; then
        echo "ERROR: Required path not found: $f"
        exit 1
    fi
done
echo "All required files found."
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# ─────────────────────────────────────────────────────────
# Run scoring
# ─────────────────────────────────────────────────────────

echo "Starting scoring_accuracy.py..."
echo ""

/WAVE/users2/unix/tkhambadkone/.conda/envs/coe/bin/python $SCRIPT \
    --extracted   $EXTRACTED \
    --problems    $PROBLEMS \
    --image_dir   $IMAGE_DIR \
    --qa_model    $QA_MODEL \
    --output      $OUTPUT \
    --seed        42

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "End time:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: scoring_accuracy.py failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "Results written to: $OUTPUT"
