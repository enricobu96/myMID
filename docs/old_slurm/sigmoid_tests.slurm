#!/bin/bash

### SLURM Directives
#SBATCH --nodelist=dellsrv2
#SBATCH --job-name="mid_sigmoid"
#SBATCH --partition=allgroups
#SBATCH --mem=48G
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:TESLA-T4-16GB:1
#SBATCH --cpus-per-task=20


echo "Running on ${SLURM_NODELIST}"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate mid
echo "UNIV"
python $HOME/projects/cosineTests/myMID/main.py --config $HOME/projects/cosineTests/myMID/configs/baseline.yaml --dataset univ
echo "HOTEL"
python $HOME/projects/cosineTests/myMID/main.py --config $HOME/projects/cosineTests/myMID/configs/baseline.yaml --dataset hotel
echo "ZARA1"
python $HOME/projects/cosineTests/myMID/main.py --config $HOME/projects/cosineTests/myMID/configs/baseline.yaml --dataset zara1
echo "ZARA2"
python $HOME/projects/cosineTests/myMID/main.py --config $HOME/projects/cosineTests/myMID/configs/baseline.yaml --dataset zara2
