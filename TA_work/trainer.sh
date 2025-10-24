#!/bin/bash
#SBATCH --job-name=short_exercise_LDM
#SBATCH --output=logs/short_exercise/job_output-%j.txt
#SBATCH --error=logs/short_exercise/job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source diffscaler.sh
export PYTHONPATH="$PROJECT_DIR"
mkdir -p logs/short_exercise/

cd /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/TA_work
which python

python slurm_script_short_exercise.py
