#!/bin/bash -l
#SBATCH -A aria_giant
#SBATCH -J giant_poc-cpu
#SBATCH -o .logs/giant-cpu-%j.out
#SBATCH -e .logs/giant-cpu-%j.err
#SBATCH -t 23:59:00 #15:30:00
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --mem=256GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=94

# activate the environment
source $HOME/repos/giant/MeltEmulation/.venv/bin/activate

srun uv run python preprocessing/HIRHAM5_reanalysis/create_slurm.py