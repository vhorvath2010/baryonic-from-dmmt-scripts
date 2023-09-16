#!/bin/bash
#SBATCH -A hive-jw254
#SBATCH -J run_python_script
#SBATCH -n 4
#SBATCH -t 4-00:00:00
#SBATCH -o report-%j.out
#SBATCH --mail-user=vhorvath3@gatech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem-per-cpu=32G

cd /storage/hive/project/phy-wise/vhorvath3/baryonic-pred-from-dm/scripts

source ~/.bashrc

srun python parallel_capture_baryonics.py
