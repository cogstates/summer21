#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import sys

from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

from jadoch.core import slurm


def compute_metrics(pred):
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "preds": preds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("-p", "--partition", default="gpu,gpu-long,gpu-large")
    parser.add_argument("-l", "--local", action="store_true")
    args = parser.parse_args()

    if sys.stdin.isatty() and not args.local:
        return slurm.sbatch(
            f"python -u {os.path.abspath(__file__)} {' '.join(sys.argv[1:])}",
            flags=dict(partition=args.partition, exclude="sn-nvda8"),
            modules=["cuda102/toolkit/10.2"],
        ).returncode

    dataset = DatasetDict.load_from_disk(args.path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenized_dataset = dataset.map(
        lambda dt: tokenizer(dt["text"], padding="max_length", truncation=True),
        batched=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=2
    )
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    results = trainer.evaluate()
    labels = ["na", "ja"]
    for idx, pred in enumerate(results["eval_preds"]):
        print(
            {
                "text": tokenized_dataset["test"][idx]["text"],
                "label": labels[tokenized_dataset["test"][idx]["label"]],
                "prediction": labels[pred],
            }
        )
    print(results)


if __name__ == "__main__":
    main()
