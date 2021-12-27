from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
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



os.chdir("/Users/john/PycharmProjects/summer21/ldccb/data/annotation/train")
train_text = []
train_labels = []
for file_annotation, file_source in zip(sorted(glob.glob("*.xml")), sorted(glob.glob("*.txt"))):
    text, labels = get_data(file_annotation, file_source)
    train_text.append(text)
    train_labels.append(labels)



train_labels = [item for sublist in train_labels for item in sublist]
train_text =  [cleanhtml(item).strip() for sublist in train_text for item in sublist]

os.chdir("/Users/john/PycharmProjects/summer21/ldccb/data/annotation/test")
test_text = []
test_labels = []
for file_annotation, file_source in zip(sorted(glob.glob("*.xml")), sorted(glob.glob("*.txt"))):
    text, labels = get_data(file_annotation, file_source)
    test_text.append(text)
    test_labels.append(labels)

test_labels = [item for sublist in test_labels for item in sublist]
test_text =  [cleanhtml(item).strip() for sublist in test_text for item in sublist]
labels_ldc = {
    'CB': 2,
    'NCB': 1,
    'ROB': 0,
    'NA': 0
}

#print(test_text, test_labels)


lu_labels_clf = {
    'Committed Belief': 2,
    'Non-Committed Belief': 1,
    'Not Applicable': 0
}
ldc_y_train = []
for i in train_labels:
    ldc_y_train.append(labels_ldc[i])


ldc_y_test = []
for i in test_labels:
    ldc_y_test.append(labels_ldc[i])

train_text = np.array(train_text)
ldc_y_train = np.array(ldc_y_train)
empty = np.where(train_text != '')[0]
train_text = list(train_text[empty])
ldc_y_train = list(ldc_y_train[empty])

test_text = np.array(test_text)
ldc_y_test = np.array(ldc_y_test)
empty = np.where(test_text != '')[0]
test_text = list(test_text[empty])
ldc_y_test = list(ldc_y_test[empty])

print(test_text[0:100])