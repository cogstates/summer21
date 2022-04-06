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
        self.errors = {}
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
            row[self.SENTENCE] = row[self.SENTENCE][1:-2].replace("\\", "").replace("`", "")

            row[self.OFFSET_INIT], row[self.OFFSET_END], success = self.calc_offsets(row[self.FILE],
                                                                                     row[self.SENTENCE_ID],
                                                                                     row[self.SENTENCE],
                                                                                     row[self.OFFSET_INIT],
                                                                                     row[self.OFFSET_END],
                                                                                     row[self.TEXT])

            if success:
                self.raw_fb_dataset.append(row)

    def calc_offsets(self, file, sent_id, raw_sentence, offset_start, offset_end, head):
        # calculating the initial offset, since the indicies are file-based and not sentence-based in the DB
        file_offset = self.initial_offsets[(file, sent_id)]
        success = True

        # ad hoc logic to adjust offsets
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

        # keeping the asterisks just for easier understanding of the error dataset
        result_sentence = raw_sentence[:offset_start] + "* " + head + " *" + raw_sentence[offset_end:]
        if pred_head != head:
            success = False
            self.errors[(file, sent_id)] = (offset_start, offset_end, pred_head, head, raw_sentence, result_sentence)

        return offset_start, offset_end, success

    def populate_database(self):
        for row in self.raw_fb_dataset:
            # inserting sentences
            self.ma_cur.execute('INSERT INTO sentences (file, file_sentence_id, sentence) VALUES (?, ?, ?);',
                                (row[self.FILE], row[self.SENTENCE_ID], row[self.SENTENCE]))

            # need to retrieve the global sentence id since the db generates it before inserting on mentions table
            global_sentence_id = self.ma_cur.execute('SELECT sentence_id FROM sentences ORDER BY sentence_id DESC LIMIT 1;').fetchone()[0]

            self.ma_cur.execute('INSERT INTO mentions (sentence_id, token_text, token_offset_start, token_offset_end) VALUES (?, ?, ?, ?);',
                                (global_sentence_id, row[self.TEXT], row[self.OFFSET_INIT], row[self.OFFSET_END]))




    def close(self):
        self.fb_con.commit()
        self.ma_con.commit()
        self.fb_con.close()
        self.ma_con.close()

if __name__ == "__main__":
    test = FB2Master()
    test.load_data()
    test.populate_database()
    print(len(test.errors))
    test.close()


