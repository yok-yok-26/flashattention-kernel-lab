#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports/correctness
./build/debug/test_flashattention --log reports/correctness/latest.log | tee reports/correctness/latest.stdout.log
