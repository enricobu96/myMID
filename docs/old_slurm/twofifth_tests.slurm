#!/bin/bash

### SLURM Directives
#SBATCH --nodelist=dellsrv4
#SBATCH --job-name="mid_twofifth"
#SBATCH --partition=allgroups
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:TESLA-T4-16GB:1


echo "Running on ${SLURM_NODELIST}"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate mid
#echo "Two fifth"
#echo "UNIV"
#python $HOME/projects/cosineTests/myMID/main.py --config $HOME/projects/cosineTests/myMID/configs/baseline_tf.yaml --dataset univ
#echo "HOTEL"
#python $HOME/projects/cosineTests/myMID/main.py --config $HOME/projects/cosineTests/myMID/configs/baseline_tf.yaml --dataset hotel
#echo "ZARA1"
#python $HOME/projects/cosineTests/myMID/main.py --config $HOME/projects/cosineTests/myMID/configs/baseline_tf.yaml --dataset zara1
echo "ZARA2"
python $HOME/projects/cosineTests/myMID/main.py --config $HOME/projects/cosineTests/myMID/configs/baseline_tf.yaml --dataset zara2
