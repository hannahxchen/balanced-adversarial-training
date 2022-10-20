DATASET="mnli"
NUM_ADV_EXAMPLES=-1
MODEL_TYPE="bert-base-uncased"
MODEL_PATH="baseline_models/bert-base-uncased-mnli"
OUTPUT_DIR="baseline_models/bert-base-uncased-mnli/attack_results"
ATTACK_TYPE="antonym"
QUERY_BUDGET=2000
SKIP_LABEL=1

python attack.py --dataset $DATASET --model_type $MODEL_TYPE --model_checkpoint_path $MODEL_PATH \
 --output_dir $OUTPUT_DIR  --attack_type $ATTACK_TYPE --parallel \
 --num_examples_for_evaluation $NUM_ADV_EXAMPLES --query_budget $QUERY_BUDGET --skip_label $SKIP_LABEL
