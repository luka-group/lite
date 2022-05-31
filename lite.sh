#!/bin/bash
# run LITE
data_dir="data/processed_data"
output_dir="output"
device=0

CUDA_VISIBLE_DEVICES=$device python3 lite.py \
                             --data_dir $data_dir \
                             --output_dir $output_dir \
                             --train_batch_size 4 \
                             --num_train_epochs 2000 \
                             --margin 0.1 \
                             --save_epochs 1 \
                             --learning_rate 1e-6 \
                             --lamb 0.05


