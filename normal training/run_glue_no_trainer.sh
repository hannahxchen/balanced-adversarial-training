MODEL_NAME_OR_PATH="bert-base-uncased"
DATASET="mnli"
OUTPUT_DIR="baseline_models/bert-base-uncased-mnli"

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --task_name $DATASET \
  --per_device_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 4 \
  --output_dir $OUTPUT_DIR \