#!/bin/tcsh

#SBATCH --job-name=cogstates.
#SBATCH --ntasks 1 --cpus-per-task 1
#SBATCH --mem=10gb
#SBATCH --partition=gpuv100
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL.
#SBATCH --mail-user=osbornty@bc.edu

source /mmfs1/data/osbornty/summer21/.env/bin/activate.csh
module load transformers/4.6.1
cd /data/osbornty/summer21
python3 main.py
 
