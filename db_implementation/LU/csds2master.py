# author: tyler osborne
# 24 december 2022

from ddl import DDL
import sqlite3
from xml2csds import XMLCorpusToCSDSCollection

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

    # initializing python data structures to mimic database schema as well as setting up database itself
    def __init__(self):
        self.sentences = {}
        self.next_sentence_id = 1

        self.mentions = []
        self.sources = []
        self.attitudes = []

        self.create_tables()
        self.ma_con = sqlite3.connect("LU_master.db")
        self.ma_cur = self.ma_con.cursor()

        self.LU = XMLCorpusToCSDSCollection(
            '2010 Language Understanding',
            'CMU')
        self.collection = self.LU.create_and_get_collection()

    def process_data(self):
        pass



    @staticmethod
    def create_tables():
        db = DDL('LU')
        db.create_tables()
        db.close()

    def close(self):
        self.ma_con.close()


if __name__ == '__main__':
    test = CSDS2Master()
    test.close()


