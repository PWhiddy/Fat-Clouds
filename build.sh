#!/bin/sh

CUDA_DIR='usr/local/cuda-9.1'

/${CUDA_DIR}/bin/nvcc fluid.cu -std=c++11 -I ${CUDA_DIR}/include -L ${CUDA_DIR}/lib64 -Wno-deprecated-gpu-targets -o run_fluid
