# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random
import json
import pickle
import copy

import numpy as np
import datasets
from datasets import Dataset, load_dataset, load_metric
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from data_util import SynonymWordSubstitude, AntonymWordSubstitude
from losses import DistanceMetric, PairwiseLoss, TripletLoss

logger = logging.getLogger(__name__)

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "snli": ("premise", "hypothesis"),
    "qqp": ("question1", "question2"),
    "mrpc": ("sentence1", "sentence2"),
}

distance_metrics = {
    "cosine": DistanceMetric.COSINE,
    "euclidean": DistanceMetric.EUCLIDEAN,
    "manhattan": DistanceMetric.MANHATTAN
}

CONTRASTIVE_LOSS_NAMES = {
    "pairwise": PairwiseLoss,
    "triplet": TripletLoss
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--similarity_threshold", default=0.8, type=float, help="The similarity constraint to be considered as synonym."
    )
    parser.add_argument(
        "--perturbation_constraint", default=100, type=int, help="The maximum size of perturbation set of each word."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Data directory that stores perturbation constraints."
    )
    parser.add_argument(
        "--distance_metric", type=str, default="cosine", help="Distance metric for contrastive loss.", 
        choices=list(distance_metrics.keys())
    )
    parser.add_argument(
        "--margin", type=float, default=1.0, help="Margin for contrastive loss."
    )
    parser.add_argument(
        "--contrastive_loss_type", type=str, default="triplet", help="Types of contrastive loss for training.",
        choices=list(CONTRASTIVE_LOSS_NAMES.keys())
    )
    parser.add_argument(
        "--contrastive_loss_weight", type=float, default=1.0, help="Weight of contrastive triplet loss." 
    )
    parser.add_argument(
        "--fickle_weight", type=float, default=1.0, help="Weight of pairwise fickle loss." 
    )
    parser.add_argument(
        "--obstinate_weight", type=float, default=1.0, help="Weight of pairwise obstinate loss." 
    )
    parser.add_argument(
        "--skip_label", type=int, default=None, help="Labels to skip during contrastive learning."
    )
    parser.add_argument(
        "--sentence_embed_type", type=str, default="cls", help="Method for getting the sentence embedding",
        choices=["cls", "mean_pool", "max_pool"]
    )
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def load_randomized_examples(args, word_sub_table, dataset, antonym=False):
    col1, col2 = task_to_keys[args.task_name]

    id_list = []
    text_a_list = []
    text_b_list = []
    mask_list = []
    label_list = []

    if not antonym:
        for example in tqdm(dataset):
            mask = 1
            text_a = word_sub_table.get_perturbed_sentence(example[col1])
            if text_a == example[col1]:
                mask = 0

            if col2:
                text_b = word_sub_table.get_perturbed_sentence(example[col2])

                if text_a == example[col1] and text_b == example[col2]:
                    mask = 0

            text_a_list.append(text_a)
            if col2:
                text_b_list.append(text_b)
            if "idx" in example:
                id_list.append(example['idx'])
            label_list.append(example["label"])
            mask_list.append(mask)

    else:
        for example in tqdm(dataset):
            if example["label"] == args.skip_label:
                text_a = example[col1]
                if col2:
                    text_b = example[col2]

                mask = 0
            else:
                # For antonym attack, sample which sentence to be perturbed
                if col2 is not None:
                    sample_col_order = random.sample([col1, col2], 2)
                    text_a, text_b = example[col1], example[col2]
                    mask = 0
                    for col in sample_col_order:
                        perturbed = word_sub_table.get_perturbed_sentence(example[col])
                        # skip examples with no antonym substitutions found
                        if perturbed != example[col]:
                            mask = 1
                            if col == col1:
                                text_a = perturbed
                            else:
                                text_b = perturbed
                            break
                else:
                    text_a = word_sub_table.get_perturbed_sentence(example[col1])
                    if text_a == example[col1]:
                        mask = 0
                    else:
                        mask = 1
                
            text_a_list.append(text_a)
            if col2:
                text_b_list.append(text_b)
            if "idx" in example:
                id_list.append(example['idx'])
            mask_list.append(mask)

    if col2:
        data = {
            col1: text_a_list,
            col2: text_b_list,
            "mask": mask_list
        }
    else:
        data = {
            col1: text_a_list,
            "mask": mask_list
        }

    if id_list:
        data["idx"] = id_list

    randomized_dataset = Dataset.from_dict(data)
    # randomized_dataset.save_to_disk(os.path.join(args.output_dir, "randomized_examples", f"train_epoch_{epoch}"))

    return randomized_dataset

def get_sentence_embedding(args, model, input_batch, token_embeddings=None):
    # Reference: sentence_transformers library

    if token_embeddings is None:
        token_embeddings = model(**input_batch, output_hidden_states=True).hidden_states[-1]

    if args.sentence_embed_type == "cls":
        sentence_embedding = token_embeddings[:, 0]

    elif args.sentence_embed_type == "mean_pool":
        #Mean Pooling - Take attention mask into account for correct averaging
        input_mask_expanded = input_batch["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sentence_embedding = sum_embeddings / sum_mask

    elif args.sentence_embed_type == "max_pool":
        # Max Pooling - Take the max value over time for every dimension.
        input_mask_expanded = input_batch["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        sentence_embedding = torch.max(token_embeddings, 1)[0]

    return sentence_embedding

def main():
    args = parse_args()
    args.n_gpu = torch.cuda.device_count()

    if args.seed is None:
        args.seed = random.randint(1, 10000)

    with open(os.path.join(args.output_dir, 'training_args.json'), 'w') as f:
        training_args = vars(copy.deepcopy(args))
        training_args["lr_scheduler_type"] = str(args.lr_scheduler_type)
        json.dump(training_args, f, indent=4, sort_keys=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Setup logging, we only want one process per machine to log things on the screen.
    # logger.setLevel(logging.ERROR)
    # datasets.utils.logging.set_verbosity_error()
    # transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        if args.task_name in ["mnli", "qqp", "mrpc"]:
            raw_datasets = load_dataset("glue", args.task_name)
        else:
            raw_datasets = load_dataset(args.task_name).filter(lambda x: x["label"] != -1)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    model.to("cuda")

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        if "mask" in examples:
            result["mask"] = examples["mask"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # # filter labels that do not work for obstinate attacks
    # filtered_train_dataset = filter_label(args, raw_datasets["train"])
    # processed_filtered_train_dataset = filtered_train_dataset.map(
    #     preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
    # )

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    # train_dataloader = DataLoader(
    #     train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    # )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size * args.n_gpu)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    train_batch_size = args.per_device_train_batch_size * args.n_gpu * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / train_batch_size)
    if args.max_train_steps is None:
        args.max_train_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Get the metric function
    if args.task_name in ["mnli", "qqp", "mrpc"]:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps, last_log_step = 0, 0
    train_nll_loss, train_contrastive_loss = 0, 0
    nll_epoch_loss, contrastive_epoch_loss = 0, 0

    # random smoother
    with open(os.path.join(args.data_dir, f"{args.task_name}_perturbation_constraint_pca{args.similarity_threshold}_{args.perturbation_constraint}.pkl"), "rb") as f:
        synonym_perturb_set = pickle.load(f)
    with open(os.path.join(args.data_dir, f"{args.task_name}_antonym_perturbation_set.pkl"), "rb") as f:
        antonym_perturb_set = pickle.load(f)

    synonym_sub_table = SynonymWordSubstitude(synonym_perturb_set)
    antonym_sub_table = AntonymWordSubstitude(antonym_perturb_set)
    contrastive_loss_fn = CONTRASTIVE_LOSS_NAMES[args.contrastive_loss_type](distance_metrics[args.distance_metric], margin=args.margin)

    for epoch in range(args.num_train_epochs):
        model.train()
        
        logger.info("Generating randomized train examples")
        shuffle_seed = random.randint(0, 10000)
        anchor_dataset = raw_datasets["train"].shuffle(shuffle_seed)
        processed_anchor_dataset = processed_datasets["train"].shuffle(shuffle_seed)

        fickle_dataset = load_randomized_examples(args, synonym_sub_table, anchor_dataset)
        obstinate_dataset = load_randomized_examples(args, antonym_sub_table, anchor_dataset , antonym=True)
        processed_fickle_dataset = fickle_dataset.map(preprocess_function, batched=True, remove_columns=fickle_dataset.column_names)
        processed_obstinate_dataset = obstinate_dataset.map(preprocess_function, batched=True, remove_columns=obstinate_dataset.column_names)

        anchor_dataloader = DataLoader(
            processed_anchor_dataset, shuffle=False, collate_fn=data_collator, batch_size=train_batch_size
        )
        fickle_dataloader = DataLoader(
            processed_fickle_dataset, shuffle=False, collate_fn=data_collator, batch_size=train_batch_size
        )
        obstinate_dataloader = DataLoader(
            processed_obstinate_dataset, shuffle=False, collate_fn=data_collator, batch_size=train_batch_size
        )
        fickle_dataloader, obstinate_dataloader = iter(fickle_dataloader), iter(obstinate_dataloader)

        # Training with contrastive loss on both types of adversarial examples
        for step, anchor_batch in enumerate(anchor_dataloader):
            fickle_batch = next(fickle_dataloader)
            obstinate_batch = next(obstinate_dataloader)
            anchor_batch = {k: v.to("cuda") for k, v in anchor_batch.items()}
            fickle_batch = {k: v.to("cuda") for k, v in fickle_batch.items()}
            obstinate_batch = {k: v.to("cuda") for k, v in obstinate_batch.items()}

            fickle_input_batch = {k: v for k, v in fickle_batch.items() if k != "mask"}
            obstinate_input_batch = {k: v for k, v in obstinate_batch.items() if k != "mask"}

            anchor_outputs = model(**anchor_batch, output_hidden_states=True)

            nll_loss = anchor_outputs.loss

            if args.n_gpu > 1:
                nll_loss = nll_loss.mean()

            train_nll_loss += nll_loss.item()
            nll_epoch_loss += nll_loss.item()

            anchor = get_sentence_embedding(args, model, anchor_batch, token_embeddings=anchor_outputs.hidden_states[-1])
            fickle = get_sentence_embedding(args, model, fickle_input_batch)
            obstinate = get_sentence_embedding(args, model, obstinate_input_batch)

            # mask out contrastive loss for skipped labels and invalid adversarial examples
            fickle_mask = fickle_batch["mask"].float()
            obstinate_mask = obstinate_batch["mask"].float()

            if args.contrastive_loss_type == "triplet":
                contrastive_loss = contrastive_loss_fn(anchor, fickle, obstinate, obstinate_mask)
                train_contrastive_loss += contrastive_loss.item()
                contrastive_epoch_loss += contrastive_loss.item()
                contrastive_loss = contrastive_loss * args.contrastive_loss_weight
            else:
                fickle_loss, obstinate_loss = contrastive_loss_fn(anchor, fickle, obstinate, fickle_mask, obstinate_mask)
                train_contrastive_loss += (fickle_loss + obstinate_loss).item()
                contrastive_epoch_loss += (fickle_loss + obstinate_loss).item()
                contrastive_loss = fickle_loss * args.fickle_weight + obstinate_loss * args.obstinate_weight

            loss = (contrastive_loss + nll_loss) / args.gradient_accumulation_steps
            loss.backward()

            if step % args.gradient_accumulation_steps == 0 or step == len(anchor_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps > 0:
                progress_bar.set_description(
                    f"Train NLL Loss {train_nll_loss/completed_steps:.5f} / Contrastive Loss {train_contrastive_loss/completed_steps:.5f}"
                )

            if completed_steps >= args.max_train_steps:
                break

        logger.info(f"epoch {epoch}: train nll loss {train_nll_loss/completed_steps:.5f} / contrastive Loss {train_contrastive_loss/completed_steps:.5f}")

        train_metric = {
            "nll_loss": nll_epoch_loss / (completed_steps - last_log_step), 
            "contrastive_loss": contrastive_epoch_loss / (completed_steps - last_log_step)
        }
        save_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "train_metric.json"), "w") as f:
            json.dump(train_metric, f, indent=4, sort_keys=True)

        last_log_step = completed_steps
        nll_epoch_loss, contrastive_epoch_loss = 0, 0

        model.eval()
        logger.info("***** Running evaluation *****")
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")

        model_to_save = (model.module if hasattr(model, "module") else model)
        model_to_save.save_pretrained(save_dir)

        with open(os.path.join(save_dir, "eval_metric.json"), "w") as f:
            json.dump(eval_metric, f, indent=4, sort_keys=True)

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset_2 = processed_datasets["validation_mismatched"]
        eval_dataloader_2 = DataLoader(
            eval_dataset_2, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )

        model.eval()
        for step, batch in enumerate(tqdm(eval_dataloader_2)):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")
        with open(os.path.join(args.output_dir, "eval_mismatched_metric.json"), "w") as f:
            json.dump(eval_metric, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()