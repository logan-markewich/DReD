#!/bin/sh

rm -rf data_cache
python train_model.py \
    --model_name_or_path t5-large \
    --do_train \
    --text_column "document" \
    --summary_column "relation" \
    --train_file dred/dred_train.csv \
    --validation_file dred/dred_val.csv \
    --output_dir trained_t5 \
    --overwrite_output_dir \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --predict_with_generate \
    --gradient_accumulation_steps=2 \
    --adafactor \
    --learning_rate 0.00005 \
    --warmup_steps 1500 \
    --cache_dir "data_cache" \
    --lr_scheduler_type "cosine" \
    --logging_steps 75 \
    --num_train_epochs 10 \
    --evaluation_strategy "steps" \
    --eval_steps 750 \
    --save_steps 750 \
    --source_prefix "prefix" \
    --disable_tqdm true
