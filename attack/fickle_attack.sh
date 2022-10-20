DATASET="mnli"
NUM_ADV_EXAMPLES=-1
MODEL_TYPE="bert-base-uncased"
WORD_SUB_TABLE="data/mnli/mnli_perturbation_constraint_pca0.8_100.pkl"
QUERY_BUDGET=2000


python safer_attack.py --dataset $DATASET --model_type $MODEL_TYPE \
--model_checkpoint_path "baseline_models/bert-base-uncased-mnli" \
--output_dir "baseline_models/bert-base-uncased-mnli/attack_results" --parallel \
--num_examples_for_evaluation $NUM_ADV_EXAMPLES --query_budget $QUERY_BUDGET  --word_sub_table_file $WORD_SUB_TABLE