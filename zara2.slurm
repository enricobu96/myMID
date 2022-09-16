#!/bin/bash

### SLURM Directives
#SBATCH --nodelist=vgpu1-0
#SBATCH --job-name="mid_zara2_baseline"
#SBATCH --partition=testing
#SBATCH --mem=32G
#SBATCH --gres=gpu:NVIDIA-RTX-A5000:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2


echo "Running on ${SLURM_NODELIST}"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate mid
python $HOME/projects/myMID/main.py --config $HOME/projects/myMID/configs/baseline.yaml --dataset zara2 | tee $HOME/projects/myMID/personal_experiments/zara2/1/raw_res
