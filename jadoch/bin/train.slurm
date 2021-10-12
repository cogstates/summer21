#!/bin/bash
#
#SBATCH --job-name=train-gpu
#SBATCH --output=/gpfs/scratch/%u/%x.%j.out
#SBATCH --ntasks-per-node=28
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --partition gpu
#SBATCH --exclude sn-nvda8

module load shared
module load cuda102/toolkit/10.2
module load anaconda/3

SCRIPT_PATH=$(scontrol show job $SLURM_JOB_ID | awk -F= '/Command=/{print $2}')
cd $(dirname $(dirname "${SCRIPT_PATH}"))

source "$(poetry env list --full-path | cut -d' ' -f1)/bin/activate"

TXT_PATH="/gpfs/scratch/$(whoami)/$SLURM_JOB_NAME.$SLURM_JOB_ID.txt"
python -u bin/train.py /gpfs/scratch/asoubki/data/english/ > $TXT_PATH
