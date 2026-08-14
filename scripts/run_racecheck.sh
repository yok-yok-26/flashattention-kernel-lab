#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p reports/racecheck
compute-sanitizer --tool racecheck --error-exitcode 1 --log-file reports/racecheck/latest.log ./build/debug/test_flashattention --log reports/correctness/racecheck_correctness.log
