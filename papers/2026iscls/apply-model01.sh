#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gpus=1
module load mamba
source activate torch
python ../apply.py \
  ../data/eval.jsonl \
  ../output01 \
  ../data \
  --batch_size 96