from collections import defaultdict, Counter
from tqdm import tqdm
from CSDS.csds import CSDS, CSDSCollection
import re
from datasets import Dataset, DatasetDict, ClassLabel, load_metric
'''
train_path = "/gpfs/scratch/jmurzaku/cogstates/conll2csds/factbank_v1/train.conll"
dev_path = "/gpfs/scratch/jmurzaku/cogstates/conll2csds/factbank_v1/dev.conll"
test_path = "/gpfs/scratch/jmurzaku/cogstates/conll2csds/factbank_v1/test.conll"
'''
train_path = '/Users/john/PycharmProjects/summer21/conll2csds/factbank_v1/train.conll'
dev_path = '/Users/john/PycharmProjects/summer21/conll2csds/factbank_v1/dev.conll'
test_path = '/Users/john/PycharmProjects/summer21/conll2csds/factbank_v1/test.conll'

class ConllCSDS(CSDS):
    head_idx = -1

    def __init__(
            self, this_text, head_idx, this_belief, this_head="",
            this_doc_id=-1, this_sentence_id=-1
    ):
        self.doc_id = this_doc_id
        self.sentence_id = this_sentence_id
        self.text = this_text
        self.head_idx = head_idx
        self.belief = this_belief
        self.head = this_head

    def get_marked_text(self):
        return self.text

def read_conll_data(path, iter=False):
    csds = CSDSCollection("")

    sentences = []
    b = []
    h = []
    with open(path, 'r') as file:
        sentence_tokens = []
        beliefs = []
        heads = []
        for line in tqdm(file):
            line = line.strip()
            array = line.split('\t')
            if len(array) < 7:
                sentences.append(sentence_tokens)
                b.append(beliefs)
                h.append(heads)
                sentence_tokens = []
                beliefs = []
                heads = []
            else:
                word = (array[1])
                belief = (array[2])
                head = int(array[4])
                sentence_tokens.append(word)
                beliefs.append(belief)
                heads.append(head)
    file.close()
    corpus = []

    for sent_index, i in enumerate(b):
        for index, ii in enumerate(i):
            if ii != '_':
                sform = sentences[sent_index]
                head_temp = sentences[sent_index][int(h[sent_index][index])]
                replacement = "* " + sentences[sent_index][int(h[sent_index][index])] + " *"
                (sform[int(h[sent_index][index])]) = replacement
                joined = ' '.join(sform)
                joined = re.sub(r'\s+([?.!,":`])', r'\1', joined)
                print(joined)
                corpus.append(
                    (joined, int(h[sent_index][index]), float(ii), sentences[sent_index][int(h[sent_index][index])]))
                (sform[int(h[sent_index][index])]) = head_temp
    for sentence_id, sample in enumerate(corpus):
        csds.add_labeled_instance(ConllCSDS(*sample, 0, sentence_id))

    text = []
    belief = []
    for instance in csds.get_next_instance():
        text.append(instance.get_marked_text())
        belief.append(instance.get_belief())

    #ataset = Dataset.from_dict({"text": text, "labels": belief})
    return text, belief
train, train_b = read_conll_data(train_path)
test, test_b = read_conll_data(test_path, iter = True)
test = test[6636:]
test_b = test_b[6636:]
train_dict = Dataset.from_dict({"text": train, "labels": train_b})
test_dict = ataset = Dataset.from_dict({"text": test, "labels": test_b})

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

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions

    r = pearsonr(labels, preds)
    mae = mean_absolute_error(labels, preds),
    return {
        'mae': mae,
        'r': r
    }



csds_datasets = hf
notify("Created dataset, now tokenizing dataset")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokenized_csds_datasets = csds_datasets.map(tokenize_function, batched=True)
notify("Done tokenizing dataset")
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 1)
notify("Starting training")
#args = TrainingArguments(num_train_epochs=1, per_device_train_batch_size=2, per_device_eval_batch_size=2, output_dir='/gpfs/scratch/jmurzaku/cogstates')
trainer = Trainer(
    model=model,
    train_dataset=tokenized_csds_datasets['train'],
    eval_dataset=tokenized_csds_datasets['eval'],
    compute_metrics=compute_metrics
)
trainer.train()
notify("Done training")
results = trainer.evaluate()
print(results)
