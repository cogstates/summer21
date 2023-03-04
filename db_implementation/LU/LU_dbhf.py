    # author: tyler osborne
# osbornty@bc.edu
# 02/23/2022

import sqlite3
from datasets import Dataset, DatasetDict, ClassLabel
import pprint

# translator class to take sqlite factbank data and output a dictionary
# containing lists of training and testing data for hugging face


class DB2HF:

    pp = pprint.PrettyPrinter()
    # static constants to index into the raw factbank dataset
    # FILE = 0
    # SENTENCE = 1
    # TOKEN_LOCATION = 2
    # TEXT = 3
    # FACT_VALUE = 4
    # OFFSET_INIT = 5
    # OFFSET_END = 6
    # SENTENCE_ID = 7

    def __init__(self):
        self.master_data_size = 0

        self.raw_LU_dataset = [] # new instance var added by tyler to hold the result of the master query
        self.training_text = []
        self.test_text = []
        self.training_labels = []
        self.test_labels = []
        self.unique_labels = []

        self.doc_id_train_or_test = {}

        self.master_query = """SELECT * FROM
             (SELECT s.file, s.file_sentence_id,
       s.sentence, m.token_text target_head, m.token_offset_start target_offset_start,
       m.token_offset_end target_offset_end, a.label
FROM attitudes a
    JOIN mentions m on m.token_id = a.target_token_id
    JOIN sentences s on m.sentence_id = s.sentence_id
    JOIN sources s2 on s2.source_id = a.source_id);"""

        self.data_count_query = """SELECT file, count(file) num_annotations FROM
             (SELECT s.file, s.file_sentence_id,
       s.sentence, m.token_text target_head, m.token_offset_start target_offset_start,
       m.token_offset_end target_offset_end, a.label
FROM attitudes a
    JOIN mentions m on m.token_id = a.target_token_id
    JOIN sentences s on m.sentence_id = s.sentence_id
    JOIN sources s2 on s2.source_id = a.source_id)

    group by file ORDER BY num_annotations desc;"""


    # connecting to the database file and executing the master query to grab all labeled instances
    def get_data(self):
        
        if __name__ == "__main__":
            db_path = 'LU_master.db'
        else:
            db_path = 'db_implementation/LU/LU_master.db'

        con = sqlite3.connect(db_path)
        cur = con.cursor()

        # training data collection, cleanup and cataloguing
        self.raw_LU_dataset = list(cur.execute(self.master_query).fetchall())
        self.master_data_size = len(self.raw_LU_dataset)

        sql_return = list(cur.execute(self.data_count_query).fetchall())
        size_so_far = 0

        i = 0
        while size_so_far < int(0.8 * self.master_data_size):
            row = sql_return[i]
            file = row[0]
            num_annotations = row[1]

            self.doc_id_train_or_test[file] = 'train'

            size_so_far += num_annotations
            i += 1
        while i < len(sql_return):
            row = sql_return[i]
            file = row[0]
            self.doc_id_train_or_test[file] = 'test'
            i += 1

    # splitting data into training and testing sets
    def populate_lists(self):

        for row in self.raw_LU_dataset:
            file = row[0]
            file_sentence_id = row[1]
            sentence = row[2]
            target_head = row[3]
            target_offset_start = row[4]
            target_offset_end = row[5]
            label = row[6]

            # Asterisk format
            # formatted_sentence = f'{sentence[:target_offset_start]}** {target_head} **{sentence[target_offset_end:]}'

            # Pipe format
            formatted_sentence = f'{target_head}|||{sentence}'


            if self.doc_id_train_or_test[file] == 'train':
                self.training_text.append(formatted_sentence)
                self.training_labels.append(label)
            else:
                self.test_text.append(formatted_sentence)
                self.test_labels.append(label)

        beliefs = []
        beliefs += self.training_labels
        beliefs += self.test_labels
        self.unique_labels = list(set(beliefs))
        self.unique_labels.sort()

        print(f'Training: {len(self.training_text)} examples, '
              f'{round(100 * (len(self.training_text) / self.master_data_size), 2)}% of all data')
        print(f'Testing: {len(self.test_text)} examples, '
              f'{round(100 * (len(self.test_text) / self.master_data_size), 2)}% of all data')
        print(f'Unique labels: {self.unique_labels}')

        self.pp.pprint(self.training_text[1:2])
        self.pp.pprint(self.training_labels[1:2])

    # returning final dictionary containing train and test data for feeding into hf
    def get_dataset_dict(self):
        self.get_data()
        self.populate_lists()

        class_label = ClassLabel(num_classes=len(self.unique_labels), names=self.unique_labels)
        train_dataset = Dataset.from_dict(
            {"text": self.training_text, "labels": list(map(class_label.str2int, self.training_labels))}
        )
        test_dataset = Dataset.from_dict(
            {"text": self.test_text, "labels": list(map(class_label.str2int, self.test_labels))}
        )
        return DatasetDict({'train': train_dataset, 'eval': test_dataset}), len(self.unique_labels)


if __name__ == "__main__":
    db2hf = DB2HF()
    db2hf.get_data()
    db2hf.populate_lists()
