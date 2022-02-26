# author: tyler osborne
# osbornty@bc.edu
# 02/23/2022

import sqlite3
from datasets import Dataset, DatasetDict, ClassLabel

# translator class to take sqlite factbank data and output a dictionary
# containing lists of training and testing data for hugging face


class DB2HF:

    # static constants to index into the raw factbank dataset
    FILE = 0
    SENTENCE = 1
    TOKEN_LOCATION = 2
    TEXT = 3
    FACT_VALUE = 4
    SPECIAL_TOKENS = ["\'s", ",", ".", "!"]

    def __init__(self):
        self.master_data_size = 0

        self.raw_fb_dataset = [] # new instance var added by tyler to hold the result of the master query
        self.training_text = []
        self.test_text = []
        self.training_labels = []
        self.test_labels = []
        self.unique_labels = []

        self.master_query = """SELECT DISTINCT s.file, s.sent, t.tokLoc, t.text, f.factValue
    FROM sentences s
    JOIN tokens_tml t
        ON s.file = t.file
               AND s.sentId = t.sentId
    JOIN fb_factValue f
        ON f.sentId = t.sentId
               AND f.eId = t.tmlTagId
               AND f.eText = t.text
               AND f.relSourceText = "'AUTHOR'"
    GROUP BY s.file, s.sentId;"""

        self.data_count_query = """SELECT count(*) from (SELECT DISTINCT *
    FROM sentences s
    JOIN tokens_tml t
        ON s.file = t.file
               AND s.sentId = t.sentId
    JOIN fb_factValue f
        ON f.sentId = t.sentId
               AND f.eId = t.tmlTagId
               AND f.eText = t.text
               AND f.relSourceText = "'AUTHOR'"
    GROUP BY s.file, s.sentId);"""

    # connecting to the database file and executing the master query to grab all labeled instances
    def get_data(self):
        con = sqlite3.connect('factbank_data.db')
        cur = con.cursor()

        sql_return = cur.execute(self.master_query)

        # data cleanup
        for row in sql_return:
            cleanRow = list(row)

            # removing enclosing single quotes
            cleanRow[self.FILE] = cleanRow[self.FILE][1:-1]
            cleanRow[self.TEXT] = cleanRow[self.TEXT][1:-1]
            cleanRow[self.FACT_VALUE] = cleanRow[self.FACT_VALUE][1:-2]

            # putting asterisks around the head
            cleanRow[self.SENTENCE] = self.do_asterisks(cleanRow[self.SENTENCE][1:-2], cleanRow[self.TEXT])

            self.raw_fb_dataset.append(cleanRow)

        # total row count of returned query
        self.master_data_size = cur.execute(self.data_count_query).fetchone()[0]

    # splitting data into training and testing sets
    def populate_lists(self):
        self.get_data()

        doc_id_count_table = {}
        for row in self.raw_fb_dataset:
            doc_id = row[self.FILE]
            if doc_id in doc_id_count_table:
                doc_id_count_table[doc_id] += 1
            else:
                doc_id_count_table[doc_id] = 1

        # designating data as test vs train
        doc_id_train_or_test = {}
        size_training = self.master_data_size - self.master_data_size // 4
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
        print('Percent of training text:', (size_training_so_far / self.master_data_size) * 100)
        if self.master_data_size != size_check_sum:
            print('Warning: size does not check out')

        for row in self.raw_fb_dataset:
            # check doc ID and train vs test
            # populate lists in loop
            doc_id = row[self.FILE]
            if doc_id_train_or_test[doc_id] == 'train':
                self.training_text.append(row[self.SENTENCE])
                self.training_labels.append(row[self.FACT_VALUE])
            else:
                self.test_text.append(row[self.SENTENCE])
                self.test_labels.append(row[self.FACT_VALUE])

        beliefs = []
        beliefs += self.training_labels
        beliefs += self.test_labels
        self.unique_labels = list(set(beliefs))
        self.unique_labels.sort()

    # adding asterisks around head of sentence
    def do_asterisks(self, raw_sentence, head):
        start = raw_sentence.index(head)
        end = start + len(head)

        result_sentence = ""
        if start != 0:
            result_sentence = raw_sentence[0:start] + "* " + head + " *" + raw_sentence[end:]
        else:
            result_sentence = "* " + head + " *" + raw_sentence[end:]

        return result_sentence

    # returning final dictionary containing train and test data for feeding into hf
    def get_dataset_dict(self):
        self.populate_lists()
        class_label = ClassLabel(num_classes=len(self.unique_labels), names=self.unique_labels)
        train_dataset = Dataset.from_dict(
            {"text": self.training_text, "labels": list(map(class_label.str2int, self.training_labels))}
        )
        test_dataset = Dataset.from_dict(
            {"text": self.test_text, "labels": list(map(class_label.str2int, self.test_labels))}
        )
        return DatasetDict({'train': train_dataset, 'eval': test_dataset})

testinstance = DB2HF()
print(testinstance.get_dataset_dict())