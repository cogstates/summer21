# author: tyler osborne
# osbornty@bc.edu
# 04/03/2021

import sqlite3
from ddl import DDL
from fb_sentence_processor import FB_SENTENCE_PROCESSOR
from progress.bar import Bar
import time

class FB2Master:
    # static constants to index into the raw factbank dataset

    FILE = 0
    SENTENCE_ID = 1
    SENTENCE = 2
    RAW_OFFSET_INIT = 2
    REL_SOURCE_TEXT = 2

    def __init__(self):

        self.start_time = time.time()

        # connecting to origin and destination database files
        self.fb_con = sqlite3.connect("factbank_data.db")
        self.fb_cur = self.fb_con.cursor()

        self.create_tables()
        self.ma_con = sqlite3.connect("fb_master.db")
        self.ma_cur = self.ma_con.cursor()

        self.initial_offsets = {}
        self.final_offsets = {}
        self.errors = {}
        self.num_errors = 0
        self.rel_source_texts = {}
        self.source_offsets = {}
        self.target_offsets = {}
        self.fact_values = {}
        self.targets = {}

        # queries to be used throughout program
        self.fb_sentences_query = """
        SELECT DISTINCT s.file, s.sentid, s.sent
        FROM sentences s
        ORDER BY s.file, s.sentid;"""
        self.offsets_query = """SELECT o.file, o.sentId, o.offsetInit FROM offsets o WHERE o.tokLoc = 0;"""

    # since FactBank's offsets are file-based, we need to convert them to sentence-based
    def load_initial_offsets(self):
        # building dictionary of initial offsets for later calculation of sentence-based token offsets
        offsets_sql_return = self.fb_cur.execute(self.offsets_query)
        for row in offsets_sql_return:
            self.initial_offsets[(row[self.FILE], row[self.SENTENCE_ID])] = row[self.RAW_OFFSET_INIT]

    # building a dictionary of every relSourceText from FactBank, mapping them to the associated sentence
    def load_rel_source_texts(self):
        rel_source_data = self.fb_cur.execute('SELECT file, sentId, relSourceText FROM fb_relSource;').fetchall()
        for row in rel_source_data:
            key = (row[self.FILE], row[self.SENTENCE_ID])
            value = str(row[self.REL_SOURCE_TEXT])[1:-2]
            if key not in self.rel_source_texts:
                self.rel_source_texts[key] = [value]
            else:
                self.rel_source_texts[key].append(value)

    # same as above except dealing with source token offsets
    def load_source_offsets(self):
        source_offsets_data = self.fb_cur.execute(
            'SELECT s.file, s.sentId, o.offsetInit, o.offsetEnd, o.text '
            'FROM fb_source s JOIN offsets o '
            'ON s.file = o.file AND s.sentId = o.sentId '
            'AND s.sourceLoc = o.tokLoc;')
        for row in source_offsets_data:
            key = (row[self.FILE], row[self.SENTENCE_ID])
            value = (row[2], row[3], str(row[4])[1:-2])
            self.source_offsets[key] = value



    # initializing the DDL for the master schema
    def create_tables(self):
        db = DDL('fb')
        db.create_tables()
        db.close()

    # retrieving every sentence and populating its tokens, sources and attitudes
    def load_data(self):

        sentences_sql_return = self.fb_cur.execute(self.fb_sentences_query).fetchall()
        sp = FB_SENTENCE_PROCESSOR(sentences_sql_return, self.initial_offsets, self.rel_source_texts,
                                   self.source_offsets, self.target_offsets, self.targets, self.fact_values)
        sp.go()
        self.errors, self.num_errors = sp.get_errors()

        # inserting python data into master schema
        self.ma_con.executemany('INSERT INTO SENTENCES (sentence_id, file, file_sentence_id, sentence) '
                                'VALUES (?, ?, ?, ?);', sp.sentences)
        self.ma_con.executemany('INSERT INTO mentions '
                                '(token_id, sentence_id, token_text, token_offset_start, token_offset_end) '
                                'VALUES (?, ?, ?, ?, ?);', sp.mentions)
        self.ma_con.executemany('INSERT INTO sources '
                                '(source_id, sentence_id, token_id, parent_source_id, nesting_level, [source]) '
                                'VALUES (?, ?, ?, ?, ?, ?);', sp.sources)
        self.ma_con.executemany('INSERT INTO attitudes '
                                '(attitude_id, source_id, target_token_id, label, label_type) '
                                'VALUES (?, ?, ?, ?, ?);', sp.attitudes)


    def load_targets(self):
        targets_raw = self.fb_cur.execute('SELECT file, sentId, tmlTagId, tokLoc, text FROM tokens_tml;').fetchall()
        for row in targets_raw:
            target_key = (row[0], row[1], row[2])
            self.targets[target_key] = [row[3], row[4]]

    def load_target_offsets(self):
        target_offsets_raw = self.fb_cur.execute('SELECT file, sentId, tokLoc, offsetInit, offsetEnd FROM offsets;').fetchall()
        for row in target_offsets_raw:
            target_offset_key = (row[0], row[1], row[2])
            self.target_offsets[target_offset_key] = [row[3], row[4]]


    def load_fact_values(self):
        fact_values_raw = self.fb_cur.execute('SELECT file, sentId, relSourceText, eId, factValue FROM fb_factValue;').fetchall()
        for row in fact_values_raw:
            fact_value_key = (row[0], row[1], row[2])
            if fact_value_key in self.fact_values:
                self.fact_values[fact_value_key].append((row[3], row[4]))
            else:
                self.fact_values[fact_value_key] = [(row[3], row[4])]

    def commit(self):
        self.fb_con.commit()
        self.ma_con.commit()

    def close(self):
        self.commit()
        self.fb_con.close()
        self.ma_con.close()

    def load_errors(self):
        print('Loading errors...')
        self.ma_cur.execute('CREATE TABLE errors ('
                            'error_id INTEGER PRIMARY KEY AUTOINCREMENT,'
                            'file VARCHAR2(255),'
                            'file_sentence_id INTEGER,'
                            'offset_start INTEGER,'
                            'offset_end INTEGER,'
                            'predicted_head VARCHAR2(255),'
                            'head VARCHAR2(255),'
                            'raw_sentence VARCHAR2(255),'
                            'result_sentence VARCHAR2(255),'
                            'rel_source_text VARCHAR2(255) )')
        bar = Bar('Errors Processed', max=len(self.errors))
        for key in self.errors:
            self.ma_cur.executemany('INSERT INTO errors (file, file_sentence_id, offset_start, '
                                    'offset_end, predicted_head, head, raw_sentence, result_sentence, rel_source_text)'
                                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', self.errors[key])
            bar.next()
        bar.finish()

    def generate_database(self):
        print("Loading Factbank data into Python data structures...")
        bar = Bar('Data Imported', max=6)
        self.load_targets()
        bar.next()
        self.load_target_offsets()
        bar.next()
        self.load_fact_values()
        bar.next()
        self.load_initial_offsets()
        bar.next()
        self.load_rel_source_texts()
        bar.next()
        self.load_source_offsets()
        bar.next()
        bar.finish()

        print('\nLoading data into master schema...')
        self.load_data()
        self.load_errors()
        self.close()
        print('Done.')

        run_time = time.time() - self.start_time
        print("Runtime:", round(run_time % 60, 3), 'sec')

if __name__ == "__main__":
    test = FB2Master()
    test.generate_database()

