# author: tyler osborne
# osbornty@bc.edu
# 04/03/2021

import sqlite3
from ddl import DDL
from os.path import exists
from os import remove


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
    REL_SOURCE_TEXT = 8

    def __init__(self):
        self.fb_con = sqlite3.connect("db_corpora/factbank_data.db")
        self.fb_cur = self.fb_con.cursor()

        self.create_tables()
        self.ma_con = sqlite3.connect("fb_master.db")
        self.ma_cur = self.ma_con.cursor()

        self.fb_master_query_author = """
            SELECT DISTINCT s.file, s.sent, t.tokLoc, t.text, f.factValue, o.offsetInit, o.offsetEnd, o.sentId, f.relSourceText
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
        self.fb_master_query_nested = """
                    SELECT DISTINCT s.file, s.sent, t.tokLoc, t.text, f.factValue, o.offsetInit, o.offsetEnd, o.sentId, f.relSourceText
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
                               AND f.relSourceText <> "'AUTHOR'"
                    GROUP BY s.file, s.sentId;"""

        self.initial_offsets = {}
        self.final_offsets = {}
        self.errors = {}
        self.offsets_query = """SELECT o.file, o.sentId, o.offsetInit FROM offsets o WHERE o.tokLoc = 0;"""
        self.raw_fb_dataset = []
        self.nesteds = {}

    def create_tables(self):
        db = DDL('fb')
        db.create_tables()
        db.close()

    def load_data(self, query):
        self.raw_fb_dataset.clear()
        # building dictionary of initial offsets for later calculation of sentence-based token offsets
        sql_return = self.fb_cur.execute(self.offsets_query)
        for row in sql_return:
            self.initial_offsets[(row[self.FILE][1:-1], row[self.SENTENCE])] = row[2]

        # training data collection, cleanup and cataloguing
        sql_return = self.fb_cur.execute(query)
        for row in sql_return:
            row = list(row)

            # removing enclosing single quotes
            row[self.FILE] = row[self.FILE][1:-1]
            row[self.TEXT] = row[self.TEXT][1:-1].replace("\\", "").replace("`", "")
            row[self.FACT_VALUE] = row[self.FACT_VALUE][1:-2]
            row[self.SENTENCE] = str(row[self.SENTENCE][1:-2].replace("\\", "").replace("`", ""))
            row[self.REL_SOURCE_TEXT] = row[self.REL_SOURCE_TEXT][1:-1]

            row[self.OFFSET_INIT], row[self.OFFSET_END], success = self.calc_offsets(row[self.FILE],
                                                                                     row[self.SENTENCE_ID],
                                                                                     row[self.SENTENCE],
                                                                                     row[self.OFFSET_INIT],
                                                                                     row[self.OFFSET_END],
                                                                                     row[self.TEXT])

            if success:
                self.raw_fb_dataset.append(row)

    def calc_nesting_level(self, source_text):
        nesting_level = source_text.count('_')
        if '=' in source_text:
            return nesting_level, source_text[0:source_text.index('=')]
        if source_text == 'AUTHOR':
            return 0, None
        return nesting_level, source_text[:source_text.index('_')]

    def calc_offsets(self, file, sent_id, raw_sentence, offset_start, offset_end, head):
        # calculating the initial offset, since the indicies are file-based and not sentence-based in the DB

        file_offset = self.initial_offsets[(file, sent_id)]
        success = True

        # ad hoc logic to adjust offsets
        head_length = offset_end - offset_start
        offset_start -= file_offset
        original_offset_start = offset_start

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
            self.errors[(file, sent_id)] = (original_offset_start, offset_start, offset_end, pred_head, head, raw_sentence, result_sentence)

        return offset_start, offset_end, success

    def populate_database(self):
        print("raw length: " + str(len(self.raw_fb_dataset)))
        for row in self.raw_fb_dataset:
            # inserting sentences
            dupe_found = self.ma_cur.execute('SELECT COUNT(*) FROM sentences WHERE sentence = ?;',
                                             [row[self.SENTENCE]]).fetchone()[0]
            if dupe_found == 0:
                self.ma_cur.execute('INSERT INTO sentences (file, file_sentence_id, sentence) VALUES (?, ?, ?);',
                                    (row[self.FILE], row[self.SENTENCE_ID], row[self.SENTENCE]))

            # need to retrieve the global sentence id since the db generates it before inserting on mentions table
            global_sentence_id = self.ma_cur.execute(
                'SELECT sentence_id FROM sentences WHERE sentence = ?;',
                (row[self.SENTENCE],)).fetchone()[0]
            self.ma_cur.execute(
                'INSERT INTO mentions (sentence_id, token_text, token_offset_start, token_offset_end) VALUES '
                '(?, ?, ?, ?);',
                (global_sentence_id, row[self.TEXT], row[self.OFFSET_INIT], row[self.OFFSET_END])
            )

            # similarly, need to retrieve the global token id since the db generates it before inserting on
            # sources table
            global_token_id = \
                self.ma_cur.execute('SELECT token_id FROM sentences s JOIN mentions m '
                                    'ON s.sentence_id = m.sentence_id WHERE s.sentence_id = ?;',
                                    (global_sentence_id,)).fetchone()[0]

            # calculating nesting level from underscore notation
            nesting_level, row[self.REL_SOURCE_TEXT] = self.calc_nesting_level(row[self.REL_SOURCE_TEXT])
            if nesting_level > 0:
                if nesting_level not in self.nesteds:
                    self.nesteds[nesting_level] = 1
                else:
                    self.nesteds[nesting_level] += 1

            self.ma_cur.execute('INSERT INTO sources (token_id, nesting_level, source) VALUES (?, ?, ?);',
                                (global_token_id, nesting_level, row[self.REL_SOURCE_TEXT]))

            # ditto on above two global ids
            global_source_id = global_token_id = \
                self.ma_cur.execute('SELECT source_id FROM sources WHERE token_id = ?;', (global_token_id,)).fetchone()[0]
            self.ma_cur.execute('INSERT INTO attitudes (source_id, target_token_id, label, label_type) VALUES '
                                '(?, ?, ?, ?);',
                                (global_source_id, global_token_id, row[self.FACT_VALUE], 'Belief'))
        self.commit()

    def commit(self):
        self.fb_con.commit()
        self.ma_con.commit()

    def close(self):
        self.commit()
        self.fb_con.close()
        self.ma_con.close()

    def findDupes(self):
        sql_return = self.ma_cur.execute("SELECT sentence FROM SENTENCES")
        items = []
        dupes = 0
        for row in sql_return:
            sentence = row[0]
            if sentence not in items:
                items.append(sentence)
            else:
                dupes += 1
        return dupes


if __name__ == "__main__":
    test = FB2Master()
    test.load_data(test.fb_master_query_author)
    test.populate_database()
    print("\n\nDUPLICATES: " + str(test.findDupes()))

    test.load_data(test.fb_master_query_nested)
    test.populate_database()
    print("\n\nDUPLICATES: " + str(test.findDupes()))
    print(test.nesteds)
    # print(len(test.raw_fb_dataset))

    f = open('errors.txt', 'w')
    f.write('##### ERROR COUNT: {0} #####\n\n\n\n\n'.format(len(test.errors)))
    for error in test.errors:
        f.write('self.errors[(file, sent_id)] = (original_offset_start,\n offset_start,\n '
                'offset_end,\n pred_head,\n head,\n raw_sentence,\n result_sentence)\n')
        f.write(str(error))
        for item in test.errors[error]:
            f.write("\n" + str(item))
        f.write('\n\n\n\n')
    f.close()
    test.close()


