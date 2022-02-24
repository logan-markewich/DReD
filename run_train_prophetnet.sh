#!/bin/sh

rm -rf data_cache
python train_model.py \
    --model_name_or_path microsoft/prophetnet-large-uncased \
    --do_train \
    --text_column "document" \
    --summary_column "relation" \
    --train_file dred_train.csv \
    --validation_file dred_val.csv \
    --output_dir trained_prophetnet \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --gradient_accumulation_steps=1 \
    --adafactor \
    --learning_rate 0.0000125 \
    --warmup_steps 1500 \
    --cache_dir "data_cache" \
    --lr_scheduler_type "cosine" \
    --logging_steps 75 \
    --num_train_epochs 10 \
    --evaluation_strategy "steps" \
    --eval_steps 750 \
    --save_steps 750 \
    --disable_tqdm true


