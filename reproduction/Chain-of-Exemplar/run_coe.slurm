#!/bin/bash
#SBATCH --job-name=coe_infer
#SBATCH --output=coe_infer_%j.out
#SBATCH --error=coe_infer_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=jloiretbernal@scu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

module load Anaconda3

export HF_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache
export TRANSFORMERS_CACHE=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache
export HUGGINGFACE_HUB_CACHE=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache/hub
export MPLCONFIGDIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/mpl_cache
export XDG_CACHE_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/xdg_cache

mkdir -p $HF_HOME
mkdir -p $HUGGINGFACE_HUB_CACHE
mkdir -p $MPLCONFIGDIR
mkdir -p $XDG_CACHE_HOME

export HF_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache
export TRANSFORMERS_CACHE=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache
export HUGGINGFACE_HUB_CACHE=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/hf_cache/hub

mkdir -p $HF_HOME
mkdir -p $HUGGINGFACE_HUB_CACHE

cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/Chain-of-Exemplar/reproduction/Chain-of-Exemplar

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu/bin/python --version

/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu/bin/python -c "import torch, transformers, peft; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available()); print('transformers', transformers.__version__); print('peft', peft.__version__)"

/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Josephine/coe_gpu/bin/python run_inference.py
