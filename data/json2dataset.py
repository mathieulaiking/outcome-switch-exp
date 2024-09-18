import os
import json
from typing import List, Dict
from datasets import Dataset, Features, Value, Sequence, ClassLabel


def parse_data_dir(json_dir, label2id) :
    ids, tokens, labels = [], [], []
    for fname in os.listdir(json_dir):
        fid = fname.split('.')[0]
        with open(os.path.join(json_dir,fname),'r') as f:
            data = json.load(f)
        for obj in data["article_bio_outcomes"]:
            ids.append(fid + "_" + obj["sentence_nb"])
            tokens.append(obj["tokens"])
            labels.append(label2id[obj["labels"]])
    return ids, tokens, labels



def get_hf_dataset(ids,tokens,labels, id2label) -> Dataset:
    """Create `datasets.Dataset` obj from BIO extracted data (sentences and labels)"""
    examples_dict = {
            'id': ids,
            'tokens': tokens,
            'labels': labels,
    }
    features = Features({
        "id" : Value('string'),
        "labels" : Sequence(ClassLabel(len(id2label),names=id2label)),
        "tokens" : Sequence(Value('string')),
    })
    return Dataset.from_dict(examples_dict, features=features)


if __name__ == "__main__":

    label2id = {
        "O": 0,
        "B-PrimaryOutcome": 1,
        "I-PrimaryOutcome": 2,
    }
    id2label = ["O", "B-PrimaryOutcome", "I-PrimaryOutcome"]
    ids,tokens,labels = parse_data_dir("data/ap-selective-reporting", label2id)