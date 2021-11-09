from collections import defaultdict, Counter
from tqdm import tqdm
from CSDS.csds import CSDS, CSDSCollection
import re
import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel, load_metric

train_path = "/gpfs/scratch/jmurzaku/cogstates/conll2csds/factbank_v1/train.conll"
dev_path = "/gpfs/scratch/jmurzaku/cogstates/conll2csds/factbank_v1/dev.conll"
test_path = "/gpfs/scratch/jmurzaku/cogstates/conll2csds/factbank_v1/test.conll"

train_path = '/Users/john/PycharmProjects/summer21/conll2csds/factbank_v1/train.conll'
dev_path = '/Users/john/PycharmProjects/summer21/conll2csds/factbank_v1/dev.conll'
test_path = '/Users/john/PycharmProjects/summer21/conll2csds/factbank_v1/test.conll'


'''
class ConllCSDS(CSDS):
Extension class to CSDS with minor changes to allow for CONLL data
'''

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
def get_spans(heads):
    spans = []
    for i in heads:
        vals, inverse, count =np.unique(i, return_inverse=True,
                                      return_counts=True)
        idx_vals_repeated = np.where(count > 1)[0]
        vals_repeated = vals[idx_vals_repeated]

        rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
        _, inverse_rows = np.unique(rows, return_index=True)
        res = np.split(cols, inverse_rows[1:])
        spans.append(res)
    return spans

def read_conll_data(data_file_path: str):
    """
    Reads Sentences and Trees from a CONLL formatted data file.

    Parameters
    ----------
    data_file_path : ``str``
        Path to data to be read.
    """
    csds = CSDSCollection("")

    sentences = []
    heads = []
    sp = []
    labels = []
    with open(data_file_path, 'r') as file:
        sentence_tokens = []
        h = []
        s = []
        for line in tqdm(file):
            line = line.strip()
            array = line.split('\t')
            if len(array) < 7:
                if sentence_tokens:
                    sentences.append(sentence_tokens)
                    sentence_tokens = []
                heads.append(h)
                labels.append(s)

                s = []
                h = []
            else:
                word = array[1]
                belief = (array[2])
                s.append(belief)
                head = int(array[4])
                pos = array[5]
                h.append(head)
                token = word
                sentence_tokens.append(token)
    corpus = []
    spans = get_spans(heads)
    bv = []
    for sent_index, i in enumerate(labels):
        spans_sent_index = spans[sent_index]
        for index, belief_value in enumerate(i):
            if belief_value != '_':

                t = []
                bvv = []
                sform = np.array(sentences[sent_index])
                for i in spans_sent_index:
                    t.append(sform[i])
                    bvv.append(belief_value)

                head_temp = sentences[sent_index][int(heads[sent_index][index])]
                replacement = "* " + sentences[sent_index][int(heads[sent_index][index])] + " *"
                (sform[int(heads[sent_index][index])]) = replacement
                joined = ' '.join(sform)
                #joined = re.sub(r'\s+([?.!,":`])', r'\1', joined)
                corpus.append(
                    (joined, int(heads[sent_index][index]), int(float(belief_value)) + 3, sentences[sent_index][int(heads[sent_index][index])]))
                (sform[int(heads[sent_index][index])]) = head_temp
                sp.append(t)
                bv.append(bvv)
    for sentence_id, sample in enumerate(corpus):
        csds.add_labeled_instance(ConllCSDS(*sample, 0, sentence_id))

    text = []
    belief = []
    for instance in csds.get_next_instance():
        text.append(instance.get_marked_text())
        belief.append(instance.get_belief())

    final = []
    final_bv = []
    for i in sp:
        for ii in i:
            final.append(' '.join(ii))

    for i in bv:
        for ii in i:
            final_bv.append(float(ii))

    return sentences, heads, final, final_bv


train, train_b, train_span, train_span_labels = read_conll_data(train_path)
test, test_b, test_span, test_span_labels = read_conll_data(test_path)

print(len(test))
train_dict = Dataset.from_dict({"text": train_span, "labels": train_span_labels})
test_dict = ataset = Dataset.from_dict({"text": test_span, "labels": test_span_labels})

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
    mae = mean_absolute_error(labels, preds),
    return {
        'mae': mae,
        'r': r
    }




csds_datasets = hf
notify("Created dataset, now tokenizing dataset")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenized_csds_datasets = csds_datasets.map(tokenize_function, batched=True)
notify("Done tokenizing dataset")
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 1)
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
#trainer.save_model("/gpfs/scratch/jmurzaku/best_models")
