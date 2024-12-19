#!/bin/bash

#$ -pe smp 4
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-1:1
#$ -N ProcessPrompts
#$ -o ./logs/
#$ -e ./logs/

TAG="12-19" # Doesn't affect performance, gets appended to output file names

# VERBOSE="False" # Concise
VERBOSE="True"

# REJECT_MODE="proportional"
REJECT_MODE="aggregate"

# DATASET="mini"
# DATASET="all"
# DATASET="tensortrust"
# DATASET="tt_extraction"
# DATASET="train_deepset"
DATASET="test_deepset"

module load conda
conda activate sse

cd /afs/crc.nd.edu/user/s/swebster/Private/SecureSoftwareEngineering
/afs/crc.nd.edu/user/s/swebster/.conda/envs/sse/bin/python3 process_prompts.py ${VERBOSE} ${TAG} ${REJECT_MODE} ${DATASET}
