# author: tyler osborne
# osbornty@bc.edu
# 03/30/2021

import sqlite3

con = sqlite3.connect('master.db')
cur = con.cursor()

# DDL for tables
def create_tables():
    cur.execute('CREATE TABLE sentences ('
                'sentence_id INTEGER PRIMARY KEY AUTOINCREMENT,'
                'file VARCHAR2(255) NOT NULL,'
                'file_sentence_id INTEGER NOT NULL,'
                'sentence VARCHAR2(255) NOT NULL )')

    cur.execute('CREATE TABLE mentions ('
                'token_id INTEGER PRIMARY KEY AUTOINCREMENT,'
                'sentence_id REFERENCES sentences(sentence_id),'
                'token_text VARCHAR2(255),'
                'token_offset_start INTEGER,'
                'token_offset_end INTEGER,'
                'phrase_text VARCHAR2(255),'
                'phrase_offset_start INTEGER,'
                'phrase_offset_end INTEGER )')


    cur.execute('CREATE TABLE sources ('
                'source_id INTEGER PRIMARY KEY AUTOINCREMENT,'
                'token_id REFERENCES mentions(token_id),'
                'nesting_level INTEGER,'
                '[source] VARCHAR2(255) )')

    cur.execute('CREATE TABLE attitudes ('
                'attitude_id INTEGER PRIMARY KEY AUTOINCREMENT,'
                'source_id REFERENCES sources(source_id),'
                'target_token_id REFERENCES mentions(token_id),'
                'label VARCHAR2(255),'
                'label_type VARCHAR2(255) )')

def clear_database():
    cur.execute('DROP TABLE attitudes')
    cur.execute('DROP TABLE sources')
    cur.execute('DROP TABLE mentions')
    cur.execute('DROP TABLE sentences')

# clear_database()
create_tables()