from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from lu2csds.db2hf import DB2HF


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
    db2hf = DB2HF()
    csds_datasets, num_labels = db2hf.get_dataset_dict()
    notify("Created dataset, now tokenizing dataset")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenized_csds_datasets = csds_datasets.map(tokenize_function, batched=True)
    notify("Done tokenizing dataset")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=num_labels)
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
