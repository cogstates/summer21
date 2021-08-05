

import datasets
from CSDS.csds import CSDS, CSDSCollection
from datasets import Dataset, DatasetDict, ClassLabel, load_metric
from xml2csds.xml2csds import XMLCorpusToCSDSCollection


class CSDS2HF:

    training_text = []
    test_text = []
    training_labels = []
    test_labels = []
    csds_collection = None

    # First create a mapping from string labels to integers
    cl = ClassLabel(num_classes=3, names=['CB', 'NCB', 'NA'])

    def __init__(self, csds_collection):
        self.csds_collection = csds_collection

    def populate_lists(self):
        text = []
        beliefs = []
        for instance in self.csds_collection.get_next_instance():
            text.append(instance.get_marked_text())
            beliefs.append(instance.get_belief())
        size = len(text)
        size_training = size - size // 4
        self.training_text = text[:size_training]
        self.test_text = text[size_training:]
        self.training_labels = beliefs[:size_training]
        self.training_labels = self.convert_labels(self.training_labels)
        self.test_labels = beliefs[size_training:]
        self.test_labels = self.convert_labels(self.test_labels)

        csds_train_dict = self.make_dict(self.training_text, self.training_labels)
        csds_eval_dict = self.make_dict(self.test_text, self.test_labels)

        print(csds_eval_dict)


    def make_dict(self, text, labels):
        csds_dict = {"text": text, "labels": map(cl.str2int, labels)}
        return csds_dict


    def convert_labels(self, labels):
        i = 0
        while i < len(labels):
            if labels[i] == "Not Applicable":
                labels[i] = "NA"
                i += 1
            elif labels[i] == "Non-Committed Belief":
                labels[i] = "NCB"
                i += 1
            else:
                labels[i] = "CB"
                i += 1
        return labels


if __name__ == '__main__':
    input_processor = XMLCorpusToCSDSCollection(
        '2010 Language Understanding',
        '../CMU')
    collection = input_processor.create_and_get_collection()
    csds2hf = CSDS2HF(collection)
    csds2hf.populate_lists()

