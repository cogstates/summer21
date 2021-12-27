#!/bin/bash
#
#SBATCH --job-name=cogstates
#SBATCH --output=res.txt
#SBATCH --ntasks-per-node=28
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH -p gpu

module load shared

module load cuda102/toolkit/10.2
cd /gpfs/home/jmurzaku/cogstates
source cogstates/bin/activate
cd /gpfs/scratch/jmurzaku/cogstates
cd conll2csds

python factbank2csds.py
