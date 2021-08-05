# Owen's experiment to convert a CSDS to the HF data structure

import datasets
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict, ClassLabel, load_metric

# create a CSDS as dict

# First create a mapping from string labels to integers
c2l = ClassLabel(num_classes=3, names=['CB', 'NCB', 'NA'])

csds_train_dict = {'text': ["John said he * likes * beets.",
                            "Mary sometimes says she likes beets.",
                            "Mary sometimes says she likes beets.",
                            "Mary sometimes says she likes beets.",
                            "Mary sometimes says she likes beets.",
                            "Mary sometimes says she likes beets.",
                            "Mary sometimes says she likes beets.",
                            "Mary sometimes says she likes beets.",
                            "Mary sometimes says she likes beets.",
                            "Mary sometimes says she likes beets.",
                            "Mary sometimes says she likes beets.",
                            "Mary sometimes says she likes beets.",
                            "Mary sometimes says she likes beets.",
                            "Mary sometimes says she likes beets.",
                            "Mary sometimes says she likes beets.",
                            "Mary sometimes says she likes beets.",
                            "Mary sometimes says she likes beets.",
                            "Mary sometimes says she likes beets.",
                            "Mary sometimes says she likes beets.",
                            "Mary maybe likes beets."
                            ],
                   'label': map(c2l.str2int, ["CB", "NCB", "NCB", "NCB", "NCB", "NCB", "NCB",
                             "NCB", "NCB", "NCB", "NCB", "NCB", "NCB", "NCB",
                             "NCB", "NCB", "NCB", "NCB", "NCB", "NCB"])}

csds_eval_dict = {'text': ["Peter said he likes beets.",
                           "Joan sometimes says she likes beets.",
                           "Joan sometimes says she likes beets.",
                           "Joan sometimes says she likes beets.",
                           "Joan sometimes says she likes beets.",
                           "Joan sometimes says she likes beets.",
                           "Joan sometimes says she likes beets.",
                           "Joan sometimes says she likes beets.",
                           "Joan sometimes says she likes beets.",
                           "Joan sometimes says she likes beets.",
                           "Joan sometimes says she likes beets.",
                           "Joan sometimes says she likes beets.",
                           "Joan sometimes says she likes beets.",
                           "Joan sometimes says she likes beets.",
                           "Joan sometimes says she likes beets.",
                           "Joan sometimes says she likes beets.",
                           "Joan sometimes says she likes beets.",
                           "Joan sometimes says she likes beets.",
                           "Joan sometimes says she likes beets.",
                           "Joan maybe likes beets."
                           ],
                  'label': map(c2l.str2int, ["CB", "NCB", "NCB", "NCB", "NCB", "NCB", "NCB",
                            "NCB", "NCB", "NCB", "NCB", "NCB", "NCB", "NCB",
                            "NCB", "NCB", "NCB", "NCB", "NCB", "NCB"])}



csds_train_dataset = Dataset.from_dict(csds_train_dict)
csds_eval_dataset = Dataset.from_dict(csds_eval_dict)
csds_datasets = DatasetDict({'train': csds_train_dataset,
                             'eval': csds_eval_dataset})


def notify(string):
    print(">>>>   ", string, "   <<<<")


notify("Created datset, now tokenizing dataset")

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_csds_datasets = csds_datasets.map(tokenize_function, batched=True)

notify("Done tokenizing dataset")

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)
metric = load_metric("accuracy")


# In the named arguments below, replace full_train_dataset
# and full-eval_dataset with small_train_dataset and
# small_eval_dataset, respectively, for experimentation with
# a small subset of the input data and a shorter running time.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


notify("Starting training")

training_args = TrainingArguments("../CSDS/test_trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_csds_datasets['train'],
    eval_dataset=tokenized_csds_datasets['eval'],
    compute_metrics=compute_metrics,
)
trainer.train()

notify("Done training")

results = trainer.evaluate()
print(results)
