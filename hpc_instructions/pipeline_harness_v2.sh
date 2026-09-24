#!/bin/bash
set -euo pipefail

# 1. Pre-flight sanity checks
COARSE_COUNT=$(ls -1 coarse_results_*.csv 2>/dev/null | wc -l)
if [ "$COARSE_COUNT" -eq 0 ]; then
    echo "Error: No coarse_results_*.csv files detected in $(pwd)."
    echo "Wait until Job 2 finishes writing results before running this harness."
    exit 1
fi
echo "Verified: $COARSE_COUNT coarse result chunks present."

# 2. Check if user passed an active Job 2 ID to wait on
COARSE_JOB_DEP=""
if [ $# -ge 1 ]; then
    COARSE_JOB_ID="$1"
    echo "Attaching to active Coarse Array Job: $COARSE_JOB_ID"
    # Ensure brackets are included for array dependency
    if [[ "$COARSE_JOB_ID" != *"[]"* ]]; then
        COARSE_JOB_ID="${COARSE_JOB_ID}[]"
    fi
    COARSE_JOB_DEP="-W depend=afterany:$COARSE_JOB_ID"
fi

# 3. Clean up any leftover temporary files from old runs
rm -f refine_tasks.csv final_sweep_results.csv

# 4. Submit Job 3 (Init Refine)
echo "Submitting Stage 3: Init Refine..."
if [ -n "$COARSE_JOB_DEP" ]; then
    JOB3=$(qsub $COARSE_JOB_DEP init_refine_v2.txt)
else
    JOB3=$(qsub init_refine_v2.txt)
fi
echo "  -> Stage 3 Job ID: $JOB3"

# 5. Submit Job 4 (Refine Array, depends on Job 3 completion)
echo "Submitting Stage 4: Refine Array..."
JOB4=$(qsub -W depend=afterok:$JOB3 refine_array_v2.txt)
echo "  -> Stage 4 Job ID: $JOB4"

# 6. Submit Job 5 (Finalize, depends on entire Refine Array)
# In PBS Pro, an array dependency requires the "[]" suffix.
echo "Submitting Stage 5: Finalize..."
JOB5=$(qsub -W depend=afterany:${JOB4} finalise_results_v2.txt)
echo "  -> Stage 5 Job ID: $JOB5"