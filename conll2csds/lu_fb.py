from collections import defaultdict, Counter

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from CSDS.csds import CSDS, CSDSCollection
import re
import pandas as pd
from ast import literal_eval
from csds2hf.csds2hf import CSDS2HF
from xml2csds.xml2csds import XMLCorpusToCSDSCollection, XMLCorpusToCSDSCollectionWithOLabels
import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel, load_metric

#rain_path = "/gpfs/scratch/jmurzaku/cogstates/conll2csds/factbank_v1/train.conll"
#dev_path = "/gpfs/scratch/jmurzaku/cogstates/conll2csds/factbank_v1/dev.conll"
#test_path = "/gpfs/scratch/jmurzaku/cogstates/conll2csds/factbank_v1/test.conll"


corpus = pd.read_csv("Labeled Sentences.csv")
text_init = (list(corpus['sentence']))
text = []
for i in text_init:
    text.append(literal_eval(i))
t = []
for i in text:
    i = ([i.strip("''") for i in i])
    i = ([i.strip("'\\'") for i in i])
    i = [i.replace("'", "") for i in i]
    i = ' '.join(i)

    t.append(i)
print(t)
labels = (list(corpus['label']))
l = []
for i in labels:
    l.append(i.strip("''\n"))

'''
Running Factbank experiments
'''
train_ratio = 0.68
validation_ratio = 0.25
test_ratio = 0.07
labels = {
    'CT+': 3.0,
    'PR+': 2.0,
    'PS+': 1.0,
    'Uu': 0.0,
    'PS-': -1.0,
    'PR-': -2.0,
    'CT-': -3.0,
    'CTu': 0.0,
    'NA': 0.0
}

lu_labels = {
    'Committed Belief': 3.0,
    'Non-Committed Belief': 2.0,
    'Not Applicable': 0.0
}

lu_labels_clf = {
    'Committed Belief': 2,
    'Non-Committed Belief':1,
    'Not Applicable': 0
}

clf = {
    'CT+': 2,
    'PR+': 1,
    'PS+': 1,
    'Uu': 0,
    'PS-': 1,
    'PR-': 1,
    'CT-': 2,
    'CTu': 0,
    'NA': 0
}

x_train, x_test, y_train, y_test = train_test_split(t, l, test_size=1 - train_ratio, shuffle=True, random_state=21)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), shuffle=True, random_state=21)


y = []
for i in y_train:
    y.append(clf[i])
y_train = y

y_t = []
for i in y_test:
    y_t.append(clf[i])
y_test = y_t


input_processor = XMLCorpusToCSDSCollection(
        '2010 Language Understanding',
        '/home/jmurzaku/cogstates/CMU')
collection = input_processor.create_and_get_collection()
csds2hf = CSDS2HF(collection)
csds_datasets = csds2hf.get_dataset_dict()

lu_train_text = csds_datasets['train']['text']
lu_train_labels = csds_datasets['train']['labels']

lu_test_text = csds_datasets['eval']['text']
lu_test_labels = csds_datasets['eval']['labels']

lu_y_train = []
for i in lu_train_labels:
    lu_y_train.append(lu_labels_clf[i])


lu_y_test = []
for i in lu_test_labels:
    lu_y_test.append(lu_labels_clf[i])

x_train = x_train + (lu_train_text)

x_test = x_test
y_train = y_train + (lu_y_train)
y_test = y_test
train_dict = Dataset.from_dict({"text": x_train, "labels": y_train})
test_dict = Dataset.from_dict({"text": x_test, "labels": y_test})
hf = DatasetDict({'train': train_dict, 'eval': test_dict})

from datasets import load_metric
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def notify(string):
    print(">>>>   ", string, "   <<<<")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import f1_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    r = pearsonr(labels, preds)
    mae = mean_absolute_error(labels, preds)
    return {
        "r:": r,
        "mae: ": mae
    }


def compute_f1(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    acc = accuracy_score(labels, preds)
    macro = f1_score(labels, preds, average='macro')
    return {
        'accuracy': acc,
        'per label f1': f1,
        'precision': precision,
        'recall': recall,
        'macro': macro
    }



csds_datasets = hf
notify("Created dataset, now tokenizing dataset")
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
tokenized_csds_datasets = csds_datasets.map(tokenize_function, batched=True)
notify("Done tokenizing dataset")
model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-base', num_labels = 3)
notify("Starting training")
#args = TrainingArguments(num_train_epochs=1, per_device_train_batch_size=2, per_device_eval_batch_size=2, output_dir='/gpfs/scratch/jmurzaku/cogstates')
trainer = Trainer(
    model=model,
    train_dataset=tokenized_csds_datasets['train'],
    eval_dataset=tokenized_csds_datasets['eval'],
    compute_metrics=compute_f1
)
trainer.train()
notify("Done training")
results = trainer.evaluate()
print(results)
#trainer.save_model("/gpfs/scratch/jmurzaku/best_models")
