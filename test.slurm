#!/bin/bash
#
#SBATCH --job-name=test-gpu
#SBATCH --output=res.txt
#SBATCH --ntasks-per-node=28
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH -p gpu

module load shared

module load cuda102/toolkit/10.2
cd /gpfs/home/jmurzaku/cogstates
source cogstates/bin/activate

python main.py
