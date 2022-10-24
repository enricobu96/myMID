#!/bin/bash

### SLURM Directives
#SBATCH --nodelist=dellcuda2
#SBATCH --job-name="mid_vl_test"
#SBATCH --partition=allgroups
#SBATCH --mem=64G
#SBATCH --gres=gpu:V100-PCIe-32GB:1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8


echo "Running on ${SLURM_NODELIST}"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate mid
# echo "obv on eth, variance learning, vl_sigmoid"
# python $HOME/projects/varianceLearningTests/myMID/main.py --config $HOME/projects/varianceLearningTests/myMID/configs/vl_sigmoid.yaml --dataset eth
echo "variance learning, vl_twofifth"
python $HOME/projects/varianceLearningTests/myMID/main.py --config $HOME/projects/varianceLearningTests/myMID/configs/vl_twofifth.yaml --dataset eth





