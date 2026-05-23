#!/bin/bash
# ROCm environment setup for AMD Renoir APU (gfx90c)
# Source this before running chatterbox or other PyTorch ROCm workloads
#
# Usage: source rocm_env.sh  (or prefix any command with the env vars)
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library
