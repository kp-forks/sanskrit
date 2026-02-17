#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gpus=1
module load mamba
source activate torch
python ../training.py \
  ../data/train.jsonl \
  ../data/test.jsonl \
  ../data/meaning-embeddings.npy \
  --pretrained_model_dir /path/to/sentence/model \
  --pretrained_model_data_dir /path/to/sentence/model/data \
  --data_dir ../data \
  --lemma_encoding static \
  --sentence_encoding full \
  --sense_encoding sensim \
  --batch_size 96 \
  --hidden_size 512 \
  --num_hidden_layers 6 \
  --num_attention_heads 8 \
  --max_seq_length 256 \
  --temperature 0 \
  --learning_rate 1e-5 \
  --epochs 20 \
  --eval_steps 1000 \
  --warmup_steps 5000 \
  --output_dir ../the/output/directory
