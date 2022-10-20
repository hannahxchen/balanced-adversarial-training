DATASET_CONFIGS = {
    "mnli": {
        "labels": 3,
        "label_names": ["entailment", "neutral", "contradiction"],
        "dataset_columns": (["premise", "hypothesis"], "label"),
        "train_split": "train",
        "dev_split": "validation_mismatched+validation_matched",
    },
    "qqp": {
        "labels": 2,
        "label_names": ["duplicate", "non_duplicate"],
        "dataset_columns": (["question1", "question2"], "label"),
        "train_split": "train",
        "dev_split": "validation",
    },
    "snli": {
        "label_names": ["entailment", "neutral", "contradiction"]
    },
    "mrpc": {
        "label_names": ["equivalent", "not_equivalent"]
    }
}