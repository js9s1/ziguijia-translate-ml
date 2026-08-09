#!/bin/bash
# ROCm environment setup for AMD Renoir APU (gfx90c)
# Source this before running chatterbox or other PyTorch ROCm workloads
#
# Usage: source rocm_env.sh  (or prefix any command with the env vars)
#
# Key findings:
#   - ROCm 7.x dropped gfx90c/gfx900 ISA support. Use PyTorch ROCm 6.3 wheel
#     from pytorch.org (pip install torch --index-url .../rocm6.3)
#   - HSA_XNACK=0 is required to avoid memory access faults on Renoir iGPU
#   - HSA_OVERRIDE_GFX_VERSION=9.0.0 makes ROCm treat gfx90c as gfx900
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export HSA_XNACK=0
export ROCBLAS_USE_HIPBLASLT=0

# libhipblaslt.so.1 was missing from /opt/rocm/lib (package files removed).
# Restored from pacman cache: /var/cache/pacman/pkg/hipblaslt-7.2.2-1-x86_64.pkg.tar.zst
export LD_LIBRARY_PATH="${HOME}/.local/lib/rocm:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
