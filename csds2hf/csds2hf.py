from datasets import Dataset, DatasetDict, ClassLabel, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from xml2csds.xml2csds import XMLCorpusToCSDSCollection

# First create a mapping from string labels to integers
cl = ClassLabel(num_classes=3, names=['CB', 'NCB', 'NA'])


class CSDS2HF:

    training_text = []
    test_text = []
    training_labels = []
    test_labels = []
    unique_labels = []
    csds_collection = None

    def __init__(self, csds_collection):
        self.csds_collection = csds_collection

    def populate_lists(self):
        text = []
        beliefs = []
        for instance in self.csds_collection.get_next_instance():
            text.append(instance.get_marked_text())
            beliefs.append(instance.get_belief())
        self.unique_labels = list(set(beliefs))
        size = len(text)
        size_training = size - size // 4
        self.training_text = text[:size_training]
        self.test_text = text[size_training:]
        self.training_labels = beliefs[:size_training]
        self.test_labels = beliefs[size_training:]

    def get_dataset_dict(self):
        self.populate_lists()
        class_label = ClassLabel(num_classes=len(self.unique_labels), names=self.unique_labels)
        csds_train_dataset = Dataset.from_dict(
            {"text": self.training_text, "labels": list(map(class_label.str2int, self.training_labels))}
        )
        csds_test_dataset = Dataset.from_dict(
            {"text": self.test_text, "labels": list(map(class_label.str2int, self.test_labels))}
        )
        return DatasetDict({'train': csds_train_dataset, 'eval': csds_test_dataset})


def notify(string):
    print(">>>>   ", string, "   <<<<")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == '__main__':
    input_processor = XMLCorpusToCSDSCollection(
        '2010 Language Understanding',
        '../CMU')
    collection = input_processor.create_and_get_collection()
    csds2hf = CSDS2HF(collection)
    csds_datasets = csds2hf.get_dataset_dict()
    notify("Created dataset, now tokenizing dataset")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenized_csds_datasets = csds_datasets.map(tokenize_function, batched=True)
    notify("Done tokenizing dataset")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    metric = load_metric("accuracy")
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
