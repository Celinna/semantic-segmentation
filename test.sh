#!/bin/bash

#SBATCH --chdir /home/yju/semantic-segmentation
#SBATCH --nodes 1
#SBATCH --cpus-per-task 20
#SBATCH --ntasks 1
#SBATCH --mem 90G
#SBATCH --time 2:00:00
#SBATCH --gres gpu:1
#SBATCH --account=vita

echo "izar $HOSTNAME"

module load gcc/8.4.0-cuda python/3.7.7
module load mvapich2

source /home/yju/venvs/hrnet/bin/activate

echo STARTING AT `date`

python val.py --cfg configs/swiss_okutama.yaml


echo FINISHED at `date`

