#!/bin/bash
# ROCm environment setup for AMD Strix Halo APU (gfx1151 / Radeon 8060S)
# Source this before running PyTorch ROCm workloads.
#
# Values come from rocm.env (shared with rocm_env.py).
#
# Usage: source rocm_env.sh  (or prefix any command with the env vars)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
. "$SCRIPT_DIR/rocm.env"
set +a
export LD_LIBRARY_PATH="${ROCM_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
