#!/bin/bash

### SLURM Directives
#SBATCH --nodelist=dellcuda0
#SBATCH --job-name="mid_hotel_baseline"
#SBATCH --partition=allgroups
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --gres=mps:V100-PCIE-16GB:100
#SBATCH --cpus-per-task=8


echo "Running on ${SLURM_NODELIST}"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate mid
python $HOME/projects/myMID/main.py --config $HOME/projects/myMID/configs/baseline.yaml --dataset hotel | tee $HOME/projects/myMID/personal_experiments/hotel/1/raw_res
