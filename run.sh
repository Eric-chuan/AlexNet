#!/bin/bash
#SBATCH -J hjc
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:1
module load anaconda3/5.3.0
python NetWork.py
