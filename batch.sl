#!/bin/tcsh

#SBATCH --job-name=cogstates.
#SBATCH --ntasks 1 --cpus-per-task 1
#SBATCH --mem=64gb
#SBATCH --partition=gpuv100
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL.
#SBATCH --mail-user=aviram@bc.edu

module load transformers
cd /mmfs1/data/aviram/cogstates/summer21
python3 main.py
 
