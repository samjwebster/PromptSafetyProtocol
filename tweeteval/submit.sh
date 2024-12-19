#!/bin/bash

#$ -pe smp 4
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-1:1
#$ -N SentimentAnalysis
#$ -o ./logs/
#$ -e ./logs/

module load conda
conda activate sse

cd /afs/crc.nd.edu/user/s/swebster/Private/SecureSoftwareEngineering/tweeteval
/afs/crc.nd.edu/user/s/swebster/.conda/envs/sse/bin/python3 evaluate_prompts.py
