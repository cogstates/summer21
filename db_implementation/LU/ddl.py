# author: tyler osborne
# osbornty@bc.edu
# 03/30/2021

import sqlite3
from os.path import exists
from os import remove


class DDL:
    def __init__(self, name):
        self.check_if_exists(name)
        self.con = sqlite3.connect(name + '_master.db')
        self.cur = self.con.cursor()

    # DDL for tables
    def create_tables(self):
        self.cur.execute('CREATE TABLE sentences ('
                         'sentence_id INTEGER PRIMARY KEY,'
                         'file VARCHAR2(255),'
                         'file_sentence_id INTEGER,'
                         'sentence VARCHAR2(255) )')

        self.cur.execute('CREATE TABLE mentions ('
                         'token_id INTEGER PRIMARY KEY,'
                         'sentence_id REFERENCES sentences(sentence_id),'
                         'token_text VARCHAR2(255),'
                         'token_offset_start INTEGER,'
                         'token_offset_end INTEGER,'
                         'phrase_text VARCHAR2(255),'
                         'phrase_offset_start INTEGER,'
                         'phrase_offset_end INTEGER )')

        self.cur.execute('CREATE TABLE sources ('
                         'source_id INTEGER PRIMARY KEY,'
                         'sentence_id REFERENCES sentences(sentence_id),'
                         'token_id REFERENCES mentions(token_id),'
                         'parent_source_id INTEGER,'
                         'nesting_level INTEGER,'
                         '[source] VARCHAR2(255) )')

        self.cur.execute('CREATE TABLE attitudes ('
                         'attitude_id INTEGER PRIMARY KEY,'
                         'source_id REFERENCES sources(source_id),'
                         'target_token_id REFERENCES mentions(token_id),'
                         'label VARCHAR2(255),'
                         'label_type VARCHAR2(255) )')

    def clear_database(self):
        self.cur.execute('DROP TABLE attitudes')
        self.cur.execute('DROP TABLE sources')
        self.cur.execute('DROP TABLE mentions')
        self.cur.execute('DROP TABLE sentences')

    def close(self):
        self.con.commit()
        self.con.close()

    def check_if_exists(self, name):
        full_name = name + "_master.db"
        if exists(full_name):
            print("Output file already exists. Overwriting...")
            remove(full_name)


if __name__ == "__main__":
    test = DDL("test")
    # test.clear_database()
    test.create_tables()
    test.close()
    print('success')
