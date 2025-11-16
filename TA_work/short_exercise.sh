#!/bin/bash
#SBATCH --job-name=vae_ldm
#SBATCH --output=logs/vae_ldm/job_output-%j.txt
#SBATCH --error=logs/vae_ldm/job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:00:00
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source diffscaler.sh

which python

python short_exercise.py