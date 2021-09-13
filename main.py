from datasets import load_metric
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from csds2hf.csds2hf import CSDS2HF
from xml2csds.xml2csds import XMLCorpusToCSDSCollection, XMLCorpusToCSDSCollectionWithOLabels

def notify(string):
    print(">>>>   ", string, "   <<<<")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


if __name__ == '__main__':
    input_processor = XMLCorpusToCSDSCollection(
        '2010 Language Understanding',
        'CMU')
    collection = input_processor.create_and_get_collection()
    csds2hf = CSDS2HF(collection)
    csds_datasets = csds2hf.get_dataset_dict()
    notify("Created dataset, now tokenizing dataset")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenized_csds_datasets = csds_datasets.map(tokenize_function, batched=True)
    notify("Done tokenizing dataset")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    notify("Starting training")
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_csds_datasets['train'],
        eval_dataset=tokenized_csds_datasets['eval'],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    notify("Done training")
    results = trainer.evaluate()
    print(results)
