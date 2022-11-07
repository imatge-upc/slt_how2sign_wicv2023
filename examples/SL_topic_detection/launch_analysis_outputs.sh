#!/bin/bash
#SBATCH -p gpi.compute
#SBATCH --time=24:00:00
#SBATCH --mem 100G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12


echo $(pwd)

python analysis_outputs.py