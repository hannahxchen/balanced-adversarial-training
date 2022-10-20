"""
Code adapted from TextAttack-A2T: https://github.com/QData/TextAttack-A2T
"""
import os
import argparse
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from textattack.datasets import HuggingFaceDataset
from textattack import TrainingArgs
from textattack.models.wrappers import HuggingFaceModelWrapper
import attack_recipes
from trainer import Trainer
from configs import DATASET_CONFIGS


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

def main(args):

    if args.dataset not in DATASET_CONFIGS:
        raise ValueError()
    dataset_config = DATASET_CONFIGS[args.dataset]
    train_dataset = load_dataset("glue", args.dataset, split="train").filter(lambda x: x["label"] != -1)
    eval_dataset = load_dataset("glue", args.dataset, split=dataset_config["dev_split"]).filter(lambda x: x["label"] != -1)

    train_dataset = HuggingFaceDataset(
        train_dataset,
        # dataset_columns=dataset_config["dataset_columns"],
        label_names=dataset_config["label_names"],
    )

    eval_dataset = HuggingFaceDataset(
        eval_dataset,
        # dataset_columns=dataset_config["dataset_columns"],
        label_names=dataset_config["label_names"],
    )

    if args.model_checkpoint_path:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint_path)
    else:
        num_labels = dataset_config["labels"]
        config = AutoConfig.from_pretrained(args.model_type, num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_type, config=config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_type, use_fast=True)
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    attack = attack_recipes.A2TYoo2021.build(model_wrapper, skip_column=args.skip_column)

    training_args = TrainingArgs(
        num_epochs=args.num_epochs,
        num_clean_epochs=args.num_clean_epochs,
        attack_epoch_interval=args.attack_epoch_interval,
        parallel=args.parallel,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accumu_steps,
        num_warmup_steps=args.num_warmup_steps,
        learning_rate=args.learning_rate,
        num_train_adv_examples=args.num_adv_examples,
        attack_num_workers_per_device=1,
        query_budget_train=200,
        checkpoint_interval_epochs=args.checkpoint_interval_epochs,
        output_dir=args.model_save_path,
        load_best_model_at_end=True,
        alpha=args.alpha,
        logging_interval_step=10,
        random_seed=args.seed,
        log_to_wandb=False
    )
    trainer = Trainer(
        model_wrapper,
        "classification",
        attack,
        train_dataset,
        eval_dataset,
        training_args,
    )
    trainer.train()


if __name__ == "__main__":
    def int_or_float(v):
        try:
            return int(v)
        except ValueError:
            return float(v)

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=765, help="Random seed")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of epochs to train")
    parser.add_argument(
        "--num_clean_epochs", 
        type=int, 
        default=1, 
        help="Number of epochs to train on the original dataset before adversarial training"
    )
    parser.add_argument(
        "--model_type", required=True, help="Type of model (e.g. bert-base-uncased, roberta-base)"
    )
    parser.add_argument(
        "--model_checkpoint_path", required=False, help="Path to the saved model checkpoint"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./saved_model",
        help="Directory to save model checkpoint.",
    )
    parser.add_argument(
        "--skip_column", type=str, default=None, help="Column to skip during attack (e.g. premise, hypothesis)"
    )
    parser.add_argument(
        "--skip_label", type=int, default=None, help="Label index to skip during attack"
    )
    parser.add_argument(
        "--num_adv_examples",
        type=int_or_float,
        default=1,
        help="Number (or percentage) of adversarial examples for training.",
    )
    parser.add_argument(
        "--attack_epoch_interval",
        type=int,
        default=1,
        help="Attack model to generate adversarial examples every N epochs.",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Run training with multiple GPUs."
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=8, help="Train batch size per gpu"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--num-warmup-steps", type=int, default=500, help="Number of warmup steps."
    )
    parser.add_argument(
        "--grad_accumu_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--checkpoint_interval_epochs",
        type=int,
        default=None,
        help="If set, save model checkpoint after every `N` epochs.",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Weight for adversarial loss"
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    main(args)