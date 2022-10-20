MODEL_NAME_OR_PATH="bert-base-uncased"
DATASET="mnli"
DATA_DIR="../data/mnli"
OUTPUT_DIR="models/bert-base-uncased-mnli"

python run_train.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --per_device_train_batch_size 8 \
  --task_name $DATASET \
  --learning_rate 3e-5 \
  --num_train_epochs 15 \
  --output_dir $OUTPUT_DIR \
  --data_dir $DATA_DIR \
  --similarity_threshold 0.8 \