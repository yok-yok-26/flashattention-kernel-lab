#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
cmake -S . -B build/debug -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_FLAGS="-G -g -lineinfo" -DFA_DEBUG_SYNC=ON
cmake --build build/debug
