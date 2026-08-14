#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports/benchmark
./build/release/bench_flashattention "$@" --csv reports/benchmark/latest.csv | tee reports/benchmark/latest.log
