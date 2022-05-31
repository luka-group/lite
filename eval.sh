#!/bin/bash
# eval model checkpoints by lite.py
# for each testing sample, fine-tuned model will rank the full typing vocab based on its confidence score
# results will be saved to modelPath/evalFileName.json
# you will need result.sh to calculate P R F-1 on the generated results

model_dir="output/yourModelName/epochsXXX"
eval_data_path="data/processed_data/dev_processed.json OR test_processed.json"
type_vocab_file="data/processed_data/types.txt"
batch_size=16
device=0

## print info
echo "Will test the $model_path on $eval_data_path"

CUDA_VISIBLE_DEVICES=$device python3 eval.py \
                             --model_dir $model_dir \
                             --eval_data_path $eval_data_path \
                             --type_vocab_file $type_vocab_file \
                             --batch $batch_size