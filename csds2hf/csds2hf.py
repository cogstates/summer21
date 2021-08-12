from datasets import Dataset, DatasetDict, ClassLabel, load_metric


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


