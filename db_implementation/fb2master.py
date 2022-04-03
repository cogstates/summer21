# author: tyler osborne
# osbornty@bc.edu
# 04/03/2021

import sqlite3


class FB2MASTER:

    def __init__(self):
        self.fb_con = sqlite3.connect("db_corpora/factbank_data.db")
        self.fb_cur = self.fb_con.cursor()

        self.ma_con = sqlite3.connect("master.db")
        self.ma_cur = self.ma_con.cursor()

        self.fb_master_query = """
            SELECT DISTINCT s.file, s.sent, t.tokLoc, t.text, f.factValue, o.offsetInit, o.offsetEnd, o.sentId
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
        self.offsets_query = """select o.file, o.sentId, o.offsetInit from offsets o where o.tokLoc = 0;"""

    def load_sentences(self):
        sql_return = self.fb_cur.execute("SELECT * FROM SENTENCES;")

        file = 0
        sent_id = 1
        sentence = 2
        for row in sql_return:
            if row[sent_id] != 0:
                self.ma_cur.execute("INSERT INTO SENTENCES (file, file_sentence_id, sentence)"
                                    "VALUES (?, ?, ?);", (row[file][1:-1],
                                                         row[sent_id],
                                                         row[sentence][1:-1].replace("\\", "")))

        self.ma_con.commit()

test = FB2MASTER()
test.load_sentences()


