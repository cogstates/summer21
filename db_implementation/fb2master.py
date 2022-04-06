# author: tyler osborne
# osbornty@bc.edu
# 04/03/2021

import sqlite3


class FB2Master:
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
        self.fb_con = sqlite3.connect("db_corpora/factbank_data.db")
        self.fb_cur = self.fb_con.cursor()

        self.ma_con = sqlite3.connect("fb_master.db")
        self.ma_cur = self.ma_con.cursor()

        self.fb_master_query = """
            SELECT DISTINCT s.file, s.sent, t.tokLoc, t.text, f.factValue, o.offsetInit, o.offsetEnd, o.sentId
            FROM sentences s
            JOIN tokens_tml t
                ON s.file = t.file
                       AND s.sentId = t.sentId
            JOIN offsets o
                on t.file = o.file
                    and t.sentId = o.sentId
                        and t.tokLoc = o.tokLoc
            JOIN fb_factValue f
                ON f.sentId = t.sentId
                       AND f.eId = t.tmlTagId
                       AND f.eText = t.text
                       AND f.relSourceText = "'AUTHOR'"
            GROUP BY s.file, s.sentId;"""

        self.initial_offsets = {}
        self.final_offsets = {}
        self.offsets_query = """SELECT o.file, o.sentId, o.offsetInit FROM offsets o WHERE o.tokLoc = 0;"""
        self.raw_fb_dataset = []

    def load_data(self):
        # building dictionary of initial offsets for later calculation of sentence-based token offsets
        sql_return = self.fb_cur.execute(self.offsets_query)
        for row in sql_return:
            self.initial_offsets[(row[self.FILE][1:-1], row[self.SENTENCE])] = row[2]

        # training data collection, cleanup and cataloguing
        sql_return = self.fb_cur.execute(self.fb_master_query)
        for row in sql_return:
            row = list(row)

            # removing enclosing single quotes
            row[self.FILE] = row[self.FILE][1:-1]
            row[self.TEXT] = row[self.TEXT][1:-1].replace("\\", "").replace("`", "")
            row[self.FACT_VALUE] = row[self.FACT_VALUE][1:-2]

            row[self.SENTENCE], success = self.calc_offsets


    def close(self):
        self.fb_con.commit()
        self.ma_con.commit()
        self.fb_con.close()
        self.ma_con.close()

if __name__ == "__main__":
    test = FB2Master()
    test.close()


