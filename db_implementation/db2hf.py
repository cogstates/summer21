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
    OFFSET_INIT = 5
    OFFSET_END = 6
    SENTENCE_ID = 7

    def __init__(self):
        self.master_data_size = 0

        self.raw_fb_dataset = [] # new instance var added by tyler to hold the result of the master query
        self.training_text = []
        self.test_text = []
        self.training_labels = []
        self.test_labels = []
        self.unique_labels = []
        self.tokens = [] # storing the non-space tokens to ensure correct indexing for head labeling

        self.initial_offsets = {}

        self.errors = {}

        self.master_query = """SELECT DISTINCT s.file, s.sent, t.tokLoc, t.text, f.factValue, o.offsetInit, o.offsetEnd, o.sentId
FROM sentences s
JOIN tokens_tml t
    ON s.file = t.file
           AND s.sentId = t.sentId
join offsets o
    on t.file = o.file
        and t.sentId = o.sentId
            and t.tokLoc = o.tokLoc
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

        # self.tokens_query = """SELECT DISTINCT text FROM tokens_tml ORDER BY text;"""

        self.offsets_query = """select o.file, o.sentId, o.offsetInit from offsets o where o.tokLoc = 0;"""

    # connecting to the database file and executing the master query to grab all labeled instances
    def get_data(self):
        con = sqlite3.connect('db_corpora/factbank_data.db')
        cur = con.cursor()

        # building dictionary of initial offsets for later use in asterisk-placing
        sql_return = cur.execute(self.offsets_query)
        for row in sql_return:
            self.initial_offsets[(row[self.FILE][1:-1], row[self.SENTENCE])] = row[2]

        # training data collection, cleanup and cataloguing
        sql_return = cur.execute(self.master_query)
        for row in sql_return:
            row = list(row)

            # removing enclosing single quotes
            row[self.FILE] = row[self.FILE][1:-1]
            row[self.TEXT] = row[self.TEXT][1:-1].replace("\\", "")
            row[self.FACT_VALUE] = row[self.FACT_VALUE][1:-2]

            # putting asterisks around the head
            row[self.SENTENCE], success = self.do_asterisks(row[self.FILE],
                                                        row[self.SENTENCE_ID],
                                                        row[self.SENTENCE][1:-2].replace("\\", ""),
                                                        row[self.TOKEN_LOCATION],
                                                        row[self.OFFSET_INIT],
                                                        row[self.OFFSET_END],
                                                        row[self.TEXT])

            if success:
                self.raw_fb_dataset.append(row)

        # total row count of returned query
        self.master_data_size = cur.execute(self.data_count_query).fetchone()[0]

        # building list of tokens
        # sql_return = cur.execute(self.tokens_query)
        #
        # for row in sql_return:
        #     row = list(row)
        #     row[0] = row[0][1:-1]
        #     self.tokens.append(row[0])

    # adding asterisks around head of sentence
    def do_asterisks(self, file, sent_id, raw_sentence, tokLoc, offset_start, offset_end, head):

        # calculating the initial offset, since the indicies are file-based and not sentence-based in the DB
        file_offset = self.initial_offsets[(file, sent_id)]
        success = True

        head_length = offset_end - offset_start
        offset_start -= file_offset
        while (
                0 < offset_start < len(raw_sentence) and
                raw_sentence[offset_start] not in ' `"'
        ):
            offset_start -= 1
        if offset_start > 0:
            offset_start += 1
        offset_end = offset_start + head_length
        pred_head = raw_sentence[offset_start:offset_end]

        result_sentence = raw_sentence[:offset_start] + "* " + head + " *" + raw_sentence[offset_end:]
        if pred_head != head:
            success = False
            self.errors[(file, sent_id)] = (offset_start, offset_end, pred_head, head, raw_sentence, result_sentence)

        return result_sentence, success

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

    def get_examples_with_head_more_than_once(self):
        results = []
        for row in self.raw_fb_dataset:
            if row[self.SENTENCE].count(row[self.TEXT]) > 1:
                results.append(row)
        return results

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
        return DatasetDict({'train': train_dataset, 'eval': test_dataset}), len(self.unique_labels)


testinstance = DB2HF()
print(testinstance.get_dataset_dict())
for key1, key2 in testinstance.errors:
    print(key1, key2)
    print(testinstance.errors[(key1, key2)])
    print("\n\n")
print(len(testinstance.errors))