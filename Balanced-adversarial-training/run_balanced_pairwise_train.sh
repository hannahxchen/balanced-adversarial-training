MODEL_NAME_OR_PATH="bert-base-uncased"
DATASET="mnli"
DATA_DIR="../data/mnli"
OUTPUT_DIR="pairwise_models/bert-base-uncased-mnli"
DISTANCE_METRIC="cosine"
SENTENCE_EMBED="cls"
LOSS_TYPE="pairwise"
SKIP_LABEL=1

python balanced_adv_train.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --task_name $DATASET  \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir $OUTPUT_DIR \
  --data_dir $DATA_DIR \
  --similarity_threshold 0.8 \
  --distance_metric $DISTANCE_METRIC \
  --margin 1.0 \
  --contrastive_loss_weight 1.0 \
  --skip_label $SKIP_LABEL \
  --sentence_embed_type $SENTENCE_EMBED \
  --contrastive_loss_type $LOSS_TYPE \
  --oversensitive_weight 1.0 \
  --undersensitive_weight 1.2 \