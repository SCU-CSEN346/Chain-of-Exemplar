#!/bin/bash
#SBATCH --job-name=coe_lora
#SBATCH --output=coe_lora_%j.out
#SBATCH --error=coe_lora_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=jloiretbernal@scu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

module load Anaconda3
unset PYTHONPATH

export HF_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache
export TRANSFORMERS_CACHE=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache
export HUGGINGFACE_HUB_CACHE=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache/hub
export MPLCONFIGDIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/mpl_cache
export XDG_CACHE_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/xdg_cache
export PIP_CACHE_DIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/pip_cache
export TMPDIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/tmp

mkdir -p $HF_HOME $HUGGINGFACE_HUB_CACHE $MPLCONFIGDIR $XDG_CACHE_HOME $PIP_CACHE_DIR $TMPDIR

cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu/bin/python --version

/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu/bin/python -c "import torch; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available())"

/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu/bin/python finetune.py \
    --model_name_or_path Qwen/Qwen-VL-Chat \
    --data_path data/coe_train.json \
    --fp16 True \
    --fix_vit True \
    --output_dir output_lora_coe_train \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 1024 \
    --lazy_preprocess True \
    --gradient_checkpointing \
    --use_lora
