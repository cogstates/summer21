#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer


def compute_metrics(pred):
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'preds': preds
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    args = parser.parse_args()

    dataset = DatasetDict.load_from_disk(args.path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenized_dataset = dataset.map(
        lambda dt: tokenizer(dt["text"], padding="max_length", truncation=True),
        batched=True
    )

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset["train"], #.select(range(100)),
        eval_dataset=tokenized_dataset["test"], #.select(range(100)),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    results = trainer.evaluate()
    labels = ["na", "ja"]
    for idx, pred in enumerate(results["eval_preds"]):
        print({
            "text": tokenized_dataset["test"][idx]["text"],
            "label": labels[tokenized_dataset["test"][idx]["label"]],
            "prediction": labels[pred]
        })
    print(results)


if __name__ == "__main__":
    main()
