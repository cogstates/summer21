from os import X_OK
import xml.etree.ElementTree as ET
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
# Annotations:
# https://docs.python.org/2/library/xml.etree.elementtree.html
# parse tree
import glob, os
import re
from datasets import load_metric
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def get_data(file_annotation, file_source):

    mytree = ET.parse(file_annotation)
    root = mytree.getroot()

    labels = []
    offsets = []
    for annotation in root.findall("annotation"):
        belief = annotation.find('belief_type').text
        # annotation_text = annotation.find('annotation_text').text
        labels.append(belief)
        offsets.append(annotation.get('offset'))

    # Text to annotation:
    # sentence, label

    with open(file_source,'r') as f:
        text = f.read()

    reverse_text = text[::-1]
    # print(reverse_text)
    length = len(reverse_text)

    txt = []
    lbls = []
    for i in range(len(offsets)):
        starting = int(offsets[i])
        # new_starting = text.rfind('.', starting)
        # new_end = text.find('.', starting)
        rstarting = length - 1 - starting
        #     # print(rstarting)
        new_starting = length - 1 - reverse_text.find('.', rstarting)
        new_end = text.find('.', starting)
        # print(new_starting)
        # print(new_end)
        sentence = text[new_starting + 2:new_end + 1]
        txt.append(sentence)
        lbls.append(labels[i])
    return txt, labels

CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  cleantext = re.sub("\n","",cleantext)
  return cleantext



os.chdir("/home/jmurzaku/cogstates/conll2csds/data/annotation/train")
train_text = []
train_labels = []
for file_annotation, file_source in zip(sorted(glob.glob("*.xml")), sorted(glob.glob("*.txt"))):
    text, labels = get_data(file_annotation, file_source)
    train_text.append(text)
    train_labels.append(labels)



train_labels = [item for sublist in train_labels for item in sublist]
train_text =  [cleanhtml(item).strip() for sublist in train_text for item in sublist]

os.chdir("/home/jmurzaku/cogstates/conll2csds/data/annotation/test")
test_text = []
test_labels = []
for file_annotation, file_source in zip(sorted(glob.glob("*.xml")), sorted(glob.glob("*.txt"))):
    text, labels = get_data(file_annotation, file_source)
    test_text.append(text)
    test_labels.append(labels)

test_labels = [item for sublist in test_labels for item in sublist]
test_text =  [cleanhtml(item).strip() for sublist in test_text for item in sublist]


labels_ldc = {
    'CB': 3,
    'NCB': 2,
    'ROB': 1,
    'NA': 0
}
ldc_y_train = []
for i in train_labels:
    ldc_y_train.append(labels_ldc[i])


ldc_y_test = []
for i in test_labels:
    ldc_y_test.append(labels_ldc[i])


train_dict = Dataset.from_dict({"text": train_text, "labels": ldc_y_train})
test_dict = Dataset.from_dict({"text": test_text, "labels": ldc_y_test})
hf = DatasetDict({'train': train_dict, 'eval': test_dict})



def notify(string):
    print(">>>>   ", string, "   <<<<")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


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
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }



csds_datasets = hf
notify("Created dataset, now tokenizing dataset")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokenized_csds_datasets = csds_datasets.map(tokenize_function, batched=True)
notify("Done tokenizing dataset")
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 4)
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
