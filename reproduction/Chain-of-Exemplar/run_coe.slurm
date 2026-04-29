#!/bin/bash
#SBATCH --job-name=coe_infer
#SBATCH --output=coe_infer_%j.out
#SBATCH --error=coe_infer_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=tkhambadkone@scu.edu
#SBATCH --mail-type=END,FAIL

module purge
module load Anaconda3
module load CUDA/12.2.1
conda activate coe

# export CUDA_HOME=/usr/local/cuda-11.8
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# which python
# python --version
# python -c "import tiktoken; print('tiktoken location:', tiktoken.__file__)"
/WAVE/users2/unix/tkhambadkone/.conda/envs/coe/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'is_available?', torch.cuda.is_available())"

cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/Chain-of-Exemplar/reproduction/Chain-of-Exemplar
/WAVE/users2/unix/tkhambadkone/.conda/envs/coe/bin/python run_inference.py
# export HF_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/hf_cache
# export TRANSFORMERS_CACHE=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/hf_cache
# export HUGGINGFACE_HUB_CACHE=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/hf_cache/hub
# export MPLCONFIGDIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/mpl_cache
# export XDG_CACHE_HOME=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/xdg_cache
# export PIP_CACHE_DIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/pip_cache
# export TMPDIR=/WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/tmp

# mkdir -p $HF_HOME
# mkdir -p $HUGGINGFACE_HUB_CACHE
# mkdir -p $MPLCONFIGDIR
# mkdir -p $XDG_CACHE_HOME
# mkdir -p $PIP_CACHE_DIR
# mkdir -p $TMPDIR

# cd /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/Chain-of-Exemplar/reproduction/Chain-of-Exemplar

# echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# nvidia-smi

# /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/coe_gpu/bin/python --version

# /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/coe_gpu/bin/python -c "import torch, transformers, peft; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available()); print('transformers', transformers.__version__); print('peft', peft.__version__)"

# /WAVE/projects/CSEN-346-Sp26/Group1/Group1_Tara/coe_gpu/bin/python run_inference.py