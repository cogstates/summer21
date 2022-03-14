from datasets import load_metric
# from dataclasses import dataclass
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from csds2hf.csds2hf import CSDS2HF
from xml2csds.xml2csds import XMLCorpusToCSDSCollection
import argparse


def notify(string):
    print(">>>>   ", string, "   <<<<")


class Experiment:
    # training_corpora = [] # selects the corpus used for training
    # test_corpora = [] # selects the corpus used for testing / evaluating
    # ml_system = []  # for now, just huggingface.
    language_model = ""  # selects the language model
    # presentation = [] #
    label_mapping = set()  # choosing which labels to track: CB, NCB, NA, O. should define externally in json files

    def __init__(self, corpora, language_model, label_mapping):
        self.corpora = corpora
        # self.ml_system = ml_system
        self.language_model = language_model
        # self.presentation = presentation
        self.label_mapping = label_mapping
        self.collection = None
        self.csds_datasets = None
        self.tokenizer = None
        self.tokenized_csds_datasets = None
        self.metric = None
        self.trainer = None

    def xml2csds(self):
        input_processor = XMLCorpusToCSDSCollection(self.corpora[0], self.corpora[1])
        self.collection = input_processor.create_and_get_collection()

    def csds2hf(self):
        csds2hf = CSDS2HF(self.collection)
        self.csds_datasets = csds2hf.get_dataset_dict()

    def tokenize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.language_model)  # language_model input here
        self.tokenized_csds_datasets = self.csds_datasets.map(self.tokenize_function, batched=True)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def train(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.language_model,
                                                                   num_labels=5)  # language_model input here
        self.metric = load_metric("accuracy", "recall")
        training_args = TrainingArguments("CSDS/test_trainer")
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.tokenized_csds_datasets['train'],
            eval_dataset=self.tokenized_csds_datasets['eval'],
            compute_metrics=self.compute_metrics,
        )
        self.trainer.train()

    def evaluate(self):
        results = self.trainer.evaluate()
        print(results)

    # Edit: option to print out logits, labels
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

# Edit: (in Experiments) pre-trained word embeddings; initializing at beginning of training
# input: file, output: embedding matrix ; numpy.load(). path to embedding matrix, etc; in argparser
# purpose: Glove, pretrained from BERT.


if __name__ == '__main__':
    print('ready to accept arguments')
    parser = argparse.ArgumentParser(description='Run Experiment')

    # Setup of experiment settings with argument parsing
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument_group('--corpus', type=str, help='corpus '
                                                              'file path')
    base_parser.add_argument('--language_model', type=str, help='language model')
    base_parser.add_argument('--label_set', type=str, help='label set', nargs='+')

    args = base_parser.parse_args()
    experiment = Experiment(
        ('2010 Language Understanding', 'CMU'),  # selects corpora
        "bert-base-cased",  # selects language model
        {'CB', 'O'}  # selects the labels to be tagged in our experiment
    )
    # start xml to csds
    # Edit: (in Experiments) hard-code corpus-dependent conversion to CSDS.
    # Edit: (in Experiments) Conditional control flow, invokes correct converter.
    #       Dictionary: corpus name -> conversion routine. don't have in main
    experiment.xml2csds()
    print("XML to CSDS done")

    # start csds to hf
    # Edit: as long as we use HF. should be in Experiments, not main().
    experiment.csds2hf()
    print("CSDS to HF done")

    # start tokenizing
    notify("Created dataset, now tokenizing dataset")
    experiment.tokenize()
    notify("Done tokenizing dataset")

    # start training
    notify("Starting training")
    experiment.train()
    notify("Done training")

    # start evaluating
    experiment.evaluate()

    # Edit: set up debug logger.
