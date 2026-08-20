#!/bin/bash

echo "Submitting Coffee Simulation Pipeline..."

# 1. Submit Coarse Init
JOB1=$(qsub init_coarse.txt)
echo "Init Coarse Job ID: $JOB1"

# 2. Submit Coarse Array (Depends on 1)
JOB2=$(qsub -W depend=afterany:$JOB1 coarse_array.txt)
echo "Coarse Array Job ID: $JOB2"

# 3. Submit Refine Init (Depends on 2)
JOB3=$(qsub -W depend=afterany:$JOB2 init_refine.txt)
echo "Init Refine Job ID: $JOB3"

# 4. Submit Refine Array (Depends on 3)
JOB4=$(qsub -W depend=afterany:$JOB3 refine_array.txt)
echo "Refine Array Job ID: $JOB4"

# 5. Submit Finalize (Depends on 4)
JOB5=$(qsub -W depend=afterany:$JOB4 finalise_results.txt)
echo "Finalize Job ID: $JOB5"

echo "All jobs submitted successfully!"