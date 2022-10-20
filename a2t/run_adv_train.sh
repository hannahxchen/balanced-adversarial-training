DATASET="qqp"
MODEL_TYPE="roberta-base"
NUM_EPOCHS=4
NUM_CLEAN_EPOCHS=1
NUM_ADV_EXAMPLES=0.2
LEARNING_RATE=5e-5
NUM_WARMUP_STEPS=5000
MODEL_SAVE_PATH="models/roberta/a2t_adv_train"
PER_GPU_BATCH_SIZE=8
ATTACK_TYPE="a2t"

python adv_train.py --dataset $DATASET --model_type $MODEL_TYPE --num_epochs $NUM_EPOCHS \
 --num_clean_epochs $NUM_CLEAN_EPOCHS --learning_rate $LEARNING_RATE --model_save_path $MODEL_SAVE_PATH \
 --per_device_train_batch_size $PER_GPU_BATCH_SIZE --attack_type $ATTACK_TYPE \
 --num_adv_examples $NUM_ADV_EXAMPLES --num-warmup-steps $NUM_WARMUP_STEPS --parallel