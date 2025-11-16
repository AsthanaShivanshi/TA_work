#!/bin/bash

export BASE_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling"
export PROJECT_DIR="${BASE_DIR}/sasthana/Downscaling/TA_work"
export ENV_DIR="${BASE_DIR}/sasthana/Downscaling/Downscaling_Models/.micromamba/envs/diffscaler"

module load micromamba
eval "$(micromamba shell hook --shell=bash)"
module load gcc/13.2.0
export LD_LIBRARY_PATH=$(dirname $(gcc --print-file-name=libstdc++.so.6)):$LD_LIBRARY_PATH

micromamba activate "$ENV_DIR"

export PROJ_LIB="$ENV_DIR/share/proj"
echo $PROJ_LIB

cd "$PROJECT_DIR"