# author: tyler osborne
# osbornty@bc.edu
# 3/30/2021

import sqlite3

con = sqlite3.connect('master.db')
cur = con.cursor()

# DDL for tables
def create_tables():
    cur.execute('CREATE TABLE SENTENCES ('
                'sentence_id INTEGER PRIMARY KEY AUTOINCREMENT,'
                'file VARCHAR2(255) NOT NULL,'
                'file_sentence_id INTEGER NOT NULL,'
                'sentence varchar2(255) NOT NULL )')

    cur.execute('CREATE TABLE MENTIONS ('
                'token_id INTEGER PRIMARY KEY AUTOINCREMENT,'
                'sentence_id REFERENCES SENTENCES(sentence_id),'
                'token_text VARCHAR2(255),'
                'token_offset_start INTEGER,'
                'token_offstart_end INTEGER,'
                'phrase_text VARCHAR2(255),'
                'phrase_offset_start INTEGER,'
                'phrase_offset_end INTEGER )')


    cur.execute('CREATE TABLE SOURCES ('
                'source_id INTEGER PRIMARY KEY AUTOINCREMENT,'
                'token_id REFERENCES MENTIONS(token_id),'
                'nesting_level INTEGER,'
                '[source] VARCHAR2(255) )')

    cur.execute('CREATE TABLE ATTITUDES ('
                'attitude_id INTEGER PRIMARY KEY AUTOINCREMENT,'
                'source_id REFERENCES SOURCES(source_id),'
                'target_token_id REFERENCES MENTIONS(token_id),'
                'label VARCHAR2(255),'
                'label_type VARCHAR2(255) )')

    print('success')

def clear_database():
    cur.execute('DROP TABLE ATTITUDES')
    cur.execute('DROP TABLE SOURCES')
    cur.execute('DROP TABLE MENTIONS')
    cur.execute('DROP TABLE SENTENCES')

# clear_database()
create_tables()