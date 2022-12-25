# author: tyler osborne
# 24 december 2022

from ddl import DDL
import sqlite3
from xml2csds import XMLCorpusToCSDSCollection
import pprint

'''
CSDS Attributes:

self.doc_id = this_doc_id
self.sentence_id = this_sentence_id
self.text = this_text
self.head_start = this_head_start
self.head_end = this_head_end
self.belief = this_belief
self.head = this_head

Accessing CSDS Objects:

for entry in collection.get_next_instance():
    print(entry.get_info_short())

'''

# a class to port CSDS objects into the unified database for the Language Understanding Corpus


class CSDS2Master:
    pp = pprint.PrettyPrinter()

    # initializing python data structures to mimic database schema as well as setting up database itself
    def __init__(self):
        self.unique_sentences = {}
        self.sentences = []
        self.next_sentence_id = 1

        self.mentions = []
        self.next_mention_id = 1

        self.sources = []
        self.next_source_id = 1

        self.attitudes = []
        self.next_attitude_id = 1

        self.create_tables()
        self.ma_con = sqlite3.connect("LU_master.db")
        self.ma_cur = self.ma_con.cursor()

        self.LU = XMLCorpusToCSDSCollection(
            '2010 Language Understanding',
            'CMU')
        self.collection = self.LU.create_and_get_collection().get_all_instances()[0]

    def populate_tables(self):

        # inserting python data into master schema
        self.ma_con.executemany('INSERT INTO SENTENCES (sentence_id, file, file_sentence_id, sentence) '
                                'VALUES (?, ?, ?, ?);', self.sentences)
        self.ma_con.executemany('INSERT INTO mentions '
                                '(token_id, sentence_id, token_text, token_offset_start, '
                                'token_offset_end, phrase_text, phrase_offset_start, phrase_offset_end) '
                                'VALUES (?, ?, ?, ?, ?, ?, ?, ?);', self.mentions)
        self.ma_con.executemany('INSERT INTO sources '
                                '(source_id, sentence_id, token_id, parent_source_id, nesting_level, [source]) '
                                'VALUES (?, ?, ?, ?, ?, ?);', self.sources)
        self.ma_con.executemany('INSERT INTO attitudes '
                                '(attitude_id, source_id, target_token_id, label, label_type) '
                                'VALUES (?, ?, ?, ?, ?);', self.attitudes)

    def process_data(self):
        # print(len(self.collection))
        for example in self.collection:

            # dealing with sentences -- each CSDS object does not necessarily contain a unique sentence
            # we use a dictionary with the following function: sentence -> sentence_id
            sentence = example.text
            if sentence in self.unique_sentences:
                sentence_id = self.unique_sentences[sentence]
            else:
                self.unique_sentences[sentence] = self.next_sentence_id
                sentence_id = self.next_sentence_id
                self.next_sentence_id += 1
                self.sentences.append([sentence_id, example.doc_id, example.sentence_id, sentence])

            # dealing with heads (i.e., mentions)
            mention_id = self.next_mention_id
            self.next_mention_id += 1
            self.mentions.append([mention_id, sentence_id, example.head,
                                  example.head_start, example.head_end, None, None, None])

            # dealing with sources -- all author
            source_id = self.next_source_id
            self.next_source_id += 1
            self.sources.append([source_id, sentence_id, mention_id, None, 0, 'AUTHOR'])

            # dealing with attitudes
            attitude_id = self.next_attitude_id
            self.next_attitude_id += 1
            self.attitudes.append([attitude_id, source_id, mention_id, example.belief, 'Belief'])


    @staticmethod
    def create_tables():
        db = DDL('LU')
        db.create_tables()
        db.close()

    def close(self):
        self.ma_con.commit()
        self.ma_con.close()


if __name__ == '__main__':
    test = CSDS2Master()
    test.process_data()
    test.populate_tables()
    test.close()


