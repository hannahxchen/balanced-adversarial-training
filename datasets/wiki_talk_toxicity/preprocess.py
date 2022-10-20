import pandas as pd

comments = pd.read_csv("orig/toxicity_annotated_comments.tsv", sep="\t")
annotations = pd.read_csv("orig/toxicity_annotations.tsv", sep="\t")

annotations = annotations.groupby('rev_id').mean().reset_index()

all_data = pd.merge(annotations, comments, on = 'rev_id')
all_data['comment'] = all_data['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
all_data['comment'] = all_data['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
all_data['label'] = all_data['toxicity'] > 0.5

all_data = all_data.rename(columns={"rev_id": "idx"})
all_data["idx"] = all_data["idx"].astype(int)
all_data = all_data[["idx", "comment", "label", "split"]]

all_data.loc[all_data.split == "train"][["idx", "comment", "label"]].to_csv("wiki_train.csv", index=False)
all_data.loc[all_data.split == "dev"][["idx", "comment", "label"]].to_csv("wiki_dev.csv", index=False)
all_data.loc[all_data.split == "test"][["idx", "comment", "label"]].to_csv("wiki_test.csv", index=False)