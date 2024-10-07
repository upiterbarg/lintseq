export CUDA_VISIBLE_DEVICES=0,1
export HF_ALLOW_CODE_EVAL=1
export PATH_TO_BASELINE_DATASET = ???
export PATH_TO_EDITSEQ_DATASET = ???

NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=512
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

TOTAL_EPOCHS=10

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --main_process_port 29500 \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    finetune.py \
    --use_flash_attn \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --tokenizer_name meta-llama/Meta-Llama-3.1-8B \
    --train_file $PATH_TO_BASELINE_DATASET \
    --max_seq_length 1024 \
    --preprocessing_num_workers 1 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.001 \
    --diff_experiment 0 \
    --weight_decay 0.01 \
    --num_train_epochs $TOTAL_EPOCHS \
    --reduce_loss sum \
    --output_dir output/llama3_8b_merged_instruct_raw_pyt_lr1e-5_redsum \
    --with_tracking \
    --report_to wandb \
    --checkpointing_steps epoch \
    --logging_steps 1 \
