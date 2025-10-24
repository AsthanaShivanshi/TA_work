#!/bin/bash

export BASE_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling"
export PROJECT_DIR="${BASE_DIR}/sasthana/Downscaling/Downscaling_Models"

module load micromamba
eval "$(micromamba shell hook --shell=bash)"
module load gcc/13.2.0
export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libstdc++.so.6)):$LD_LIBRARY_PATH

# Activate the diffscaler environment
micromamba activate "$PROJECT_DIR/.micromamba/envs/diffscaler"


# PROJ_LIB for geo libraries
export PROJ_LIB="$PROJECT_DIR/.micromamba/envs/diffscaler/share/proj"
echo $PROJ_LIB

cd "$PROJECT_DIR"