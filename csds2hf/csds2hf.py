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

    # splits data into train and test
    def populate_lists(self):
        # checks example sentences in each document
        doc_id_count_table = {}
        size = self.csds_collection.get_num_labeled_instances() + self.csds_collection.get_o_instances_length()

        for instance in self.csds_collection.get_next_instance():
            # add get_doc_id in CSDS
            doc_id = instance.get_doc_id()
            if doc_id in doc_id_count_table:
                doc_id_count_table[doc_id] += 1
            else:
                doc_id_count_table[doc_id] = 1

        # assigns train or testing to the documents
        doc_id_train_or_test = {}
        size_training = size - size // 4
        size_training_so_far = 0
        size_check_sum = 0
        for document in doc_id_count_table:
            if size_training_so_far < size_training:
                doc_id_train_or_test[document] = 'train'
                size_training_so_far += doc_id_count_table[document]
            else:
                doc_id_train_or_test[document] = 'test'
            size_check_sum += doc_id_count_table[document]
        print('Size of training corpus:', size_training_so_far)
        print('Percent of training text:', (size_training_so_far / size)*100)
        if size != size_check_sum:
            print('Warning: size does not check out')

        for instance in self.csds_collection.get_next_instance():
            # check doc ID and train vs test
            # populate lists in loop
            doc_id = instance.get_doc_id()
            if doc_id_train_or_test[doc_id] == 'train':
                self.training_text.append(instance.get_marked_text())
                self.training_labels.append(instance.get_belief())
            else:
                self.test_text.append(instance.get_marked_text())
                self.test_labels.append(instance.get_belief())

        beliefs = []
        beliefs += self.training_labels
        beliefs += self.test_labels
        self.unique_labels = list(set(beliefs))

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


