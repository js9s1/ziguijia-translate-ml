#!/bin/bash
# ROCm environment setup for AMD Renoir APU (gfx90c)
# Source this before running chatterbox or other PyTorch ROCm workloads
#
# Usage: source rocm_env.sh  (or prefix any command with the env vars)
#
# Key findings:
#   - ROCm 7.x dropped gfx90c/gfx900 ISA support. Must use PyTorch ROCm 6.2 wheel
#     from pytorch.org (pip install torch==2.5.1+rocm6.2 --index-url .../rocm6.2)
#   - HSA_XNACK=0 is required to avoid memory access faults on Renoir iGPU
#   - HSA_OVERRIDE_GFX_VERSION=9.0.0 makes ROCm treat gfx90c as gfx900
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export HSA_XNACK=0
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library
