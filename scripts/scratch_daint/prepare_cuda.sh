#!/bin/bash -l

module load daint-gpu
module load cudatoolkit/9.2.148_3.19-6.0.7.1_2.1__g3d9acc8

NVCC_PATH=$(which nvcc)
CUDA_PATH=$(echo $NVCC_PATH | sed -e "s/\/bin\/nvcc//g")
export CUDA_HOME=$CUDA_PATH
export export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH