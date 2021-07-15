# imdb.py
# Script containing all the steps of the Hugging Face Transformers fine-tuning tutorial.
# Run thus in a virtual environment containing PyTorch, Transformers, Datasets, and sklearn.
# $ python imdb.py
# For the full data set, with GPUs available, this should take about 2 hours and 45 minutes to complete.
# The smaller data set (see comments below) takes only about 7 minutes and 20 seconds to complete.
# Author: Amittai Aviram - aviram@bc.edu

from datasets import load_dataset, load_metric
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

print("Loadinhg imdb dataset")

raw_datasets = load_dataset("imdb")

print("Tokenizing dataset")

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

print("Done tokenizing datasets")

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
metric = load_metric("accuracy")


# In the named arguments below, replace full_train_dataset
# and full-eval_dataset with small_train_dataset and
# small_eval_dataset, respectively, for experimentation with
# a small subset of the input data and a shorter running time.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


print("Starting training")

training_args = TrainingArguments("test_trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=full_train_dataset,
    eval_dataset=full_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
print("Done training")
results = trainer.evaluate()
print(results)
