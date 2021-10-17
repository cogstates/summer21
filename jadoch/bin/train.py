#!/usr/bin/env python
# -*- coding: utf-8 -*-
import transformers
from datasets import DatasetDict

from jadoch.core.app import harness, slurmify


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


def main(ctx):
    ctx.parser.add_argument("path")
    ctx.parser.set_defaults(
        sb_partition="gpu,gpu-long,gpu-large",
        sb_exclude="sn-nvda8",
        sb_time="08:00:00",
        modules=["cuda102/toolkit/10.2"],
    )
    args = slurmify(ctx.parser)
    # Set up logging.
    transformers.logging.disable_default_handler()
    transformers.logging.enable_propagation()
    # Tokenize the data.
    dataset = DatasetDict.load_from_disk(args.path)
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    tokenized_dataset = dataset.map(
        lambda dt: tokenizer(dt["text"], padding="max_length", truncation=True),
        batched=True,
    )
    # Train the model.
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=2
    )
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    results = trainer.evaluate()
    ctx.log.info(results)
    # Print labels if verbose.
    if not args.verbose:
        return 0
    labels = ["na", "ja"]
    for idx, pred in enumerate(results["eval_preds"]):
        ctx.log.debug(
            {
                "text": tokenized_dataset["test"][idx]["text"],
                "label": labels[tokenized_dataset["test"][idx]["label"]],
                "prediction": labels[pred],
            }
        )


if __name__ == "__main__":
    harness(main)
