#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p reports/memcheck
compute-sanitizer --tool memcheck --error-exitcode 1 --log-file reports/memcheck/latest.log ./build/debug/test_flashattention --log reports/correctness/memcheck_correctness.log
