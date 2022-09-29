#!/bin/bash

### SLURM Directives
#SBATCH --nodelist=vgpu1-0
#SBATCH --job-name="mid_testset_baseline"
#SBATCH --partition=testing
#SBATCH --mem=32G
#SBATCH --gres=gpu:NVIDIA-RTX-A5000:1
#SBATCH --time=48:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2


echo "Running on ${SLURM_NODELIST}"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate mid
# echo "TESTING ETH"
# python $HOME/projects/tests/myMID/main.py --config $HOME/projects/tests/myMID/configs/baseline.yaml --dataset eth
# echo "TESTING HOTEL"
# python $HOME/projects/tests/myMID/main.py --config $HOME/projects/tests/myMID/configs/baseline.yaml --dataset hotel
# echo "TESTING UNIV"
# python $HOME/projects/tests/myMID/main.py --config $HOME/projects/tests/myMID/configs/baseline.yaml --dataset univ
# echo "TESTING ZARA1"
# python $HOME/projects/tests/myMID/main.py --config $HOME/projects/tests/myMID/configs/baseline.yaml --dataset zara1
# echo "TESTING ZARA2"
# python $HOME/projects/tests/myMID/main.py --config $HOME/projects/tests/myMID/configs/baseline.yaml --dataset zara2
echo "TESTING STANFORD"
python $HOME/projects/tests/myMID/main.py --config $HOME/projects/tests/myMID/configs/baseline.yaml --dataset sdd
