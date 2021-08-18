from datasets import load_metric
import numpy as np

from csds2hf.csds2hf import CSDS2HF
from xml2csds.xml2csds import XMLCorpusToCSDSCollection


def notify(string):
    print(">>>>   ", string, "   <<<<")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == '__main__':
    input_processor = XMLCorpusToCSDSCollection(
        '2010 Language Understanding',
        'CMU')
    collection = input_processor.create_and_get_collection()
    csds2hf = CSDS2HF(collection)
    csds_datasets = csds2hf.get_dataset_dict()
    notify("Created dataset, now tokenizing dataset")

