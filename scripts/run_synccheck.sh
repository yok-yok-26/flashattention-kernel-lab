#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p reports/synccheck
compute-sanitizer --tool synccheck --error-exitcode 1 --log-file reports/synccheck/latest.log ./build/debug/test_flashattention --log reports/correctness/synccheck_correctness.log
