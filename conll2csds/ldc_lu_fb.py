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
import glob, os
import re
from datasets import load_metric
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

corpus = pd.read_csv("Labeled Sentences.csv")
text_init = (list(corpus['sentence']))
text = []
for i in text_init:
    text.append(literal_eval(i))
t = []
for i in text:
    i = ([i.strip("''") for i in i])
    i = ([i.strip("'\\'") for i in i])
    i = ' '.join(i)

    t.append(i)

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
    'PR+': 1.0,
    'PS+': 1.0,
    'Uu': 0.0,
    'PS-': -1.0,
    'PR-': -1.0,
    'CT-': -3.0,
    'CTu': 0.0,
    'NA': 0.0
}
lu_labels_clf = {
    'Committed Belief': 3,
    'Non-Committed Belief': 2,
    'Not Applicable': 0
}
labels_ldc = {
    'CB': 3,
    'NCB': 2, #maybe this happemned
    'ROB': 0,
    'NA': 0
}

clf = {
    'CT+': 3,
    'PR+': 2,
    'PS+': 2,
    'Uu': 0,
    'PS-': 2,
    'PR-': 2,
    'CT-': 3,
    'CTu': 0,
    'NA': 0
}

x_train, x_test, y_train, y_test = train_test_split(t, l, test_size=1 - train_ratio, shuffle=True)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), shuffle=True)


y = []
for i in y_train:
    y.append(clf[i])
y_train = y

y_t = []
for i in y_test:
    y_t.append(clf[i])
y_test = y_t

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
    'NCB': 1,
    'ROB': 0,
    'NA': 0
}

ldc_y_train = []
for i in train_labels:
    ldc_y_train.append(labels_ldc[i])


ldc_y_test = []
for i in test_labels:
    ldc_y_test.append(labels_ldc[i])

#LU
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

x_train = x_train + (lu_train_text) + train_text
x_test = x_test + (lu_test_text) + test_text

y_train = y_train + (lu_y_train) + ldc_y_train
y_test = y_test + (lu_y_test) + ldc_y_test

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
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
tokenized_csds_datasets = csds_datasets.map(tokenize_function, batched=True)
notify("Done tokenizing dataset")
model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels = 4)
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
