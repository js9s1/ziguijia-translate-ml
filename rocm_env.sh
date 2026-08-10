#!/bin/bash
# ROCm environment setup for AMD Strix Halo APU (gfx1151 / Radeon 8060S)
# Source this before running PyTorch ROCm workloads
#
# Usage: source rocm_env.sh  (or prefix any command with the env vars)
#
# Key findings:
#   - ROCm 7.2.x has native support for gfx1151 — no GFX override needed
#   - Strix Halo iGPU uses UMA (unified memory), no VRAM limits to worry about
export LD_LIBRARY_PATH="/opt/rocm/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
