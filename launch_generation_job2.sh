#!/bin/bash

#SBATCH --project=laion                                                                                                                                  
#SBATCH --account=laion 
#SBATCH --partition=g40423
#SBATCH --job-name=oa-retrieval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --mem=64GB
#SBATCH --output=%x_%j.out

source /fsx/home-lkc/env/doc2query/bin/activate
cd /fsx/home-lkc/doc2query-experiments

srun --account laion --comment laion python3 generate_queries_main.py > /fsx/home-lkc/std.out
