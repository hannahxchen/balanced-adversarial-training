"""
Code adapted from TextAttack-A2T: https://github.com/QData/TextAttack-A2T
"""

import os
import json
import random
import argparse
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from textattack.datasets import HuggingFaceDataset
from textattack import Attacker, AttackArgs
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.metrics.attack_metrics import (
    AttackQueries,
    AttackSuccessRate,
    WordsPerturbed,
)
from textattack.metrics.quality_metrics import Perplexity, USEMetric

import attack_recipes
from configs import DATASET_CONFIGS

ATTACK_RECIPE_NAMES = {
    "a2t": attack_recipes.A2TYoo2021,
    "negation": attack_recipes.NegationAttack,
    "antonym": attack_recipes.AntonymAttack,
}


def filter_fn(x):
    """Filter bad samples."""
    if x["label"] == -1:
        return False
    if "premise" in x:
        if x["premise"] is None or x["premise"] == "":
            return False
    if "hypothesis" in x:
        if x["hypothesis"] is None or x["hypothesis"] == "":
            return False
    return True


def log_attack_stats(args, results, datasplit, enable_advance_metrics=False):
    total_attacks = len(results)
    if total_attacks == 0:
        return
    
    attack_success_stats = AttackSuccessRate().calculate(results)
    words_perturbed_stats = WordsPerturbed().calculate(results)
    attack_query_stats = AttackQueries().calculate(results)
    log = {
        "Number of successful attacks": attack_success_stats["successful_attacks"],
        "Number of failed attacks": attack_success_stats["failed_attacks"],
        "Number of skipped attacks": attack_success_stats["skipped_attacks"],
        "Original accuracy": attack_success_stats["original_accuracy"],
        "Accuracy under attack": attack_success_stats["attack_accuracy_perc"],
        "Attack success rate": attack_success_stats["attack_success_rate"],
        "Average perturbed word %": words_perturbed_stats["avg_word_perturbed_perc"],
        "Average num. words per input": words_perturbed_stats["avg_word_perturbed_perc"],
        "Avg num queries": attack_query_stats["avg_num_queries"]
    }

    if enable_advance_metrics:
        perplexity_stats = Perplexity().calculate(results)
        use_stats = USEMetric().calculate(results)
        log["Average original perplexity"] = perplexity_stats["avg_original_perplexity"]
        log["Average attack perplexity"] = perplexity_stats["avg_attack_perplexity"]
        log["Average attack USE score"] = use_stats["avg_attack_use_score"]

    if args.skip_column:
        savefile = os.path.join(args.output_dir, f"{args.attack_type}_{datasplit}_eval_metrics-skip-{args.skip_column}.json")
    else:
        savefile = os.path.join(args.output_dir, f"{args.attack_type}_{datasplit}_eval_metrics.json")
    with open(savefile, "w") as f:
        f.write(json.dumps(log) + "\n")



def attack(args, datasplit="validation"):
    if args.dataset == "wiki-toxic":
        dataset = load_dataset("csv", data_files={"validation": "../datasets/wiki_talk_toxicity/wiki_dev.csv"})["validation"]

        if args.attack_type == "word_swap_toxic":
            idxs = dataset["idx"]
            random.Random(768).shuffle(idxs)
            data = dataset[:10000]
            dataset = Dataset.from_dict(data)

        label_list = ["non-toxic", "toxic"]
        label_map = {label: i for i, label in enumerate(label_list)}
        dataset = HuggingFaceDataset(dataset, dataset_columns=(("comment",), "label"), label_names=label_list, label_map=label_map)
    else:
        dataset_config = DATASET_CONFIGS[args.dataset]
        if args.dataset in ["mnli", "qqp", "mrpc"]:
                dataset = load_dataset("glue", args.dataset, split=datasplit)
        else:
            dataset = load_dataset(args.dataset, split=datasplit).filter(lambda x: x["label"] != -1)
        label_list = dataset_config["label_names"]

        dataset = HuggingFaceDataset(dataset, label_names=label_list)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, use_fast=True)
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    if args.attack_type in ATTACK_RECIPE_NAMES:
        if args.attack_type == "a2t-mlm":
            attack = ATTACK_RECIPE_NAMES[args.attack_type].build(model_wrapper, mlm=True, skip_column=args.skip_column)
        elif args.skip_label is not None:    
            attack = ATTACK_RECIPE_NAMES[args.attack_type].build(model_wrapper, skip_column=args.skip_column, skip_label=args.skip_label)
        else:
            attack = ATTACK_RECIPE_NAMES[args.attack_type].build(model_wrapper, skip_column=args.skip_column)
    else:
        raise ValueError(f"Unknown attack {args.attack_type}")

    if datasplit != "validation":
        csvfile = os.path.join(args.output_dir, f"{args.attack_type}_{datasplit}_attack_results")
    else:
        csvfile=os.path.join(args.output_dir, f"{args.attack_type}_attack_results")

    if args.skip_column:
        csvfile += f"-skip-{args.skip_column}.csv"
    else:
        csvfile += ".csv"

    attack_args = AttackArgs(
        num_examples=args.num_examples_for_evaluation, 
        query_budget=args.query_budget, 
        random_seed=args.seed, 
        parallel=args.parallel,
        disable_stdout=True,
        log_to_csv=csvfile,
        csv_coloring_style=args.csv_coloring_style
    )
    attacker = Attacker(attack, dataset, attack_args)
    attack_results = attacker.attack_dataset()

    log_attack_stats(args, attack_results, datasplit)


def main(args):

    # if args.dataset not in DATASET_CONFIGS:
    #     raise ValueError()
    
    if args.dataset == "mnli":
        datasplits = []
        if not args.skip_matched:
            datasplits.append("validation_matched")
        if not args.skip_mismatched:
            datasplits.append("validation_mismatched")

        for dataplit in datasplits:
            attack(args, dataplit)

    else:
        attack(args)

if __name__ == "__main__":
    def int_or_float(v):
        try:
            return int(v)
        except ValueError:
            return float(v)

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=765, help="Random seed")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--parallel", action="store_true", help="Run attack on multiple GPUs")
    parser.add_argument(
        "--query_budget", type=int, default=2000, 
        help="maximum number of model queries allowed per example attacked"
    )
    parser.add_argument(
        "--model_type", required=True, help="Type of model (e.g. bert-base-uncased, roberta-base)"
    )
    parser.add_argument(
        "--model_checkpoint_path", required=True, help="Path to the saved model checkpoint"
    )
    parser.add_argument(
        "--attack_type", type=str, required=True, help="Attack type: [faster-alzantot|clare]"
    )
    parser.add_argument(
        "--skip_column", type=str, default=None, help="Column to skip during attack (e.g. premise, hypothesis)"
    )
    parser.add_argument(
        "--skip_label", type=int, default=None, help="Label index (int) to skip during attack"
    )
    parser.add_argument(
        "--num_examples_for_evaluation",
        type=int_or_float,
        help="Number of adversarial examples for evaluation.",
    )
    parser.add_argument(
        "--csv_coloring_style", type=str, default="file", 
        help="Method for marking perturbed tokens [file|plain|html]"
    )
    parser.add_argument(
        "--enable_advance_metrics", action="store_true", help="Enable advance evaluation metrics (e.g. perplexity, USE score)"
    )
    parser.add_argument(
        "--skip_mismatched", action="store_true", help="Skip evaluation on MNLI validation mismatched set."
    )
    parser.add_argument(
        "--skip_matched", action="store_true", help="Skip evaluation on MNLI validation matched set."
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)