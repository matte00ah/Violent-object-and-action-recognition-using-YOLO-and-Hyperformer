#!/bin/bash
#SBATCH --job-name=Hyper3_noYolo
#SBATCH --output=results%j.txt
#SBATCH --error=output%j.txt
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2024
#SBATCH --time=02:00:00
#SBATCH --constraint="gpu_RTXA5000_24G|gpu_RTX6000_24G|gpu_A40_48G|gpu_L40S_48G"

#gpu_RTXA5000_24G|gpu_RTX6000_24G|

source $(conda info --base)/etc/profile.d/conda.sh
conda activate myenv
# conda info --envs

# to run: sbatch exec.sh

#./finetune.sh
#./finetune_Hyper2.sh
#./finetune_Hyper3.sh
./evaluate.sh

#python ./cosimo.py