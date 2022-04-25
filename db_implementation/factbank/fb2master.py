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
    RAW_OFFSET_INIT = 2

    def __init__(self):
        self.fb_con = sqlite3.connect("factbank_data.db")
        self.fb_cur = self.fb_con.cursor()

        self.create_tables()
        self.ma_con = sqlite3.connect("fb_master.db")
        self.ma_cur = self.ma_con.cursor()

        self.fb_master_query_author = """
            SELECT DISTINCT s.file, s.sent, t.tokLoc, t.text, f.factValue, o.offsetInit, o.offsetEnd, o.sentId,
                f.relSourceText
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
                       AND f.relSourceText = "'AUTHOR'";"""
        self.fb_master_query_nested_source = """
                    SELECT DISTINCT s.file, s.sent, t.tokLoc, t.text, f.factValue, o.offsetInit, o.offsetEnd, o.sentId,
                        f.relSourceText
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
                               AND f.relSourceText <> "'AUTHOR'";"""
        self.dupes_query = """
                    SELECT s.sentence, m.token_text, m.token_offset_start,
                    m.token_offset_end, s.nesting_level, s.source, a.label FROM sentences s
                    JOIN mentions m
                        ON s.sentence_id = m.sentence_id
                    JOIN sources s
                        ON m.token_id = s.token_id
                    JOIN attitudes a 
                        ON m.token_id = a.target_token_id 
                        AND s.source_id = a.source_id
                    ORDER BY s.file, s.file_sentence_id;"""

        self.initial_offsets = {}
        self.final_offsets = {}
        self.errors = {}
        self.num_errors = 0
        self.fixed_errors = {}
        self.offsets_query = """SELECT o.file, o.sentId, o.offsetInit FROM offsets o WHERE o.tokLoc = 0;"""
        self.fb_dataset = []
        self.nesteds = {}

    def create_tables(self):
        db = DDL('fb')
        db.create_tables()
        db.close()

    def load_data(self, query):
        self.fb_dataset.clear()
        # building dictionary of initial offsets for later calculation of sentence-based token offsets
        sql_return = self.fb_cur.execute(self.offsets_query)
        for row in sql_return:
            self.initial_offsets[(row[self.FILE][1:-1], row[self.SENTENCE])] = row[self.RAW_OFFSET_INIT]

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

            # adjusting offsets for remaining of incorrect alignment not caught by ad-hoc logic
            # self.fixed_errors[(file, file_sentence_id, head, rel_source_text, fact_value)] = (offset_start, offset_end)
            # error_key = (row[self.FILE], row[self.SENTENCE_ID], row[self.TEXT],
                         # row[self.REL_SOURCE_TEXT], row[self.FACT_VALUE])

            row[self.OFFSET_INIT], row[self.OFFSET_END], success = self.calc_offsets(row[self.FILE],
                                                                                     row[self.SENTENCE_ID],
                                                                                     row[self.SENTENCE],
                                                                                     row[self.OFFSET_INIT],
                                                                                     row[self.OFFSET_END],
                                                                                     row[self.TEXT],
                                                                                     row[self.REL_SOURCE_TEXT],
                                                                                     row[self.FACT_VALUE])
            if success:
                self.fb_dataset.append(row)

    def calc_nesting_level(self, source_text):
        nesting_level = source_text.count('_')
        if '=' in source_text:
            return nesting_level, source_text[0:source_text.index('=')]
        if source_text == 'AUTHOR':
            return 0, None
        return nesting_level, source_text[:source_text.index('_')]

    def calc_offsets(self, file, sent_id, raw_sentence, offset_start, offset_end, head, rel_source_text, fact_value):
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
        # self.fixed_errors[(file, file_sentence_id, head, rel_source_text, fact_value)] = (offset_start, offset_end)

        if pred_head != head and raw_sentence.count(head) == 1:
            # attempting index method if head exists uniquely in sentence
            new_offset_start = raw_sentence.index(head)
            new_offset_end = new_offset_start + head_length
            new_pred_head = raw_sentence[new_offset_start:new_offset_end]
            if new_pred_head == head:
                offset_start = new_offset_start
                offset_end = new_offset_end
            else:
                success = False
        if not success:
            self.num_errors += 1
            error_key = (file, sent_id)
            entry = (offset_start, offset_end, pred_head, head,
                     raw_sentence, result_sentence, rel_source_text, fact_value)
            if error_key not in self.errors:
                self.errors[error_key] = [entry]
            else:
                self.errors[error_key].append(entry)

        return offset_start, offset_end, success

    def populate_database(self):
        print("raw length: " + str(len(self.fb_dataset)))
        for row in self.fb_dataset:
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

            dupe_found = self.ma_cur.execute('SELECT COUNT(*) FROM mentions m '
                                             'JOIN sources s ON m.token_id = s.token_id '
                                             'JOIN attitudes a ON s.source_id = a.source_id '
                                             'AND a.target_token_id = m.token_id '
                                             'WHERE m.sentence_id = ? AND m.token_text = ? '
                                             'AND m.token_offset_start = ? AND m.token_offset_end = ? '
                                             'AND s.source = ? AND a.label = ?',
                                             (global_sentence_id, row[self.TEXT],
                                              row[self.OFFSET_INIT], row[self.OFFSET_END],
                                              row[self.REL_SOURCE_TEXT], row[self.FACT_VALUE])).fetchone()[0]
            if dupe_found == 0:
                self.ma_cur.execute(
                    'INSERT INTO mentions (sentence_id, token_text, token_offset_start, token_offset_end) VALUES '
                    '(?, ?, ?, ?);',
                    (global_sentence_id, row[self.TEXT], row[self.OFFSET_INIT], row[self.OFFSET_END])
                )

            # similarly, need to retrieve the global token id since the db generates it before inserting on
            # sources table
            global_token_id = \
                self.ma_cur.execute('SELECT m.token_id FROM mentions m '
                                    'WHERE sentence_id = ? AND m.token_text = ? '
                                    'AND m.token_offset_start = ? AND m.token_offset_end = ?;',
                                    (global_sentence_id, row[self.TEXT],
                                     row[self.OFFSET_INIT], row[self.OFFSET_END])).fetchone()[0]

            # calculating nesting level from underscore notation
            nesting_level, row[self.REL_SOURCE_TEXT] = self.calc_nesting_level(row[self.REL_SOURCE_TEXT])
            if nesting_level not in self.nesteds:
                self.nesteds[nesting_level] = 1
            else:
                self.nesteds[nesting_level] += 1

            dupe_found = self.ma_cur.execute('SELECT COUNT(*) FROM mentions m JOIN sources s '
                                             'ON m.token_id = s.token_id '
                                             'WHERE m.token_id = ? AND m.token_text = ? '
                                             'AND s.source = ? AND s.nesting_level = ?;',
                                             (global_token_id, row[self.TEXT],
                                              row[self.REL_SOURCE_TEXT], nesting_level)).fetchone()[0]
            if dupe_found == 0:
                self.ma_cur.execute('INSERT INTO sources (token_id, nesting_level, source) VALUES (?, ?, ?);',
                                    (global_token_id, nesting_level, row[self.REL_SOURCE_TEXT]))

            global_source_id = self.ma_cur.execute('SELECT source_id FROM sources WHERE token_id = ?;',
                                                   (global_token_id,)).fetchone()[0]

            dupe_found = self.ma_cur.execute('SELECT COUNT(*) FROM mentions m '
                                             'JOIN attitudes a ON m.token_id = a.target_token_id '
                                             'WHERE a.label = ? AND a.source_id = ?;',
                                             (row[self.FACT_VALUE], global_source_id)).fetchone()[0]
            if dupe_found == 0:
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
        sql_return = self.ma_cur.execute(self.dupes_query)
        items = {}
        dupes = 0
        for row in sql_return:
            data = ''
            for item in row:
                data += str(item)
            if data not in items:
                items[data] = 1
            else:
                dupes += 1
        return dupes

    def generate_error_txt(self):
        f = open('errors.txt', 'w')
        f.write('##### ERROR COUNT: {0} #####\n\n\n\n\n'.format(self.num_errors))
        for error in self.errors:
            f.write('self.errors[(file, sent_id)] = (\n offset_start,\n '
                    'offset_end,\n pred_head,\n head,\n raw_sentence,\n result_sentence,\n '
                    'rel_source_text,\n fact_value\n)')
            f.write(str(error))
            for sentence_list in self.errors[error]:
                for item in sentence_list:
                    f.write("\n" + str(item))
                f.write('\n\n\n\n')
        f.close()


if __name__ == "__main__":
    test = FB2Master()
    # test.populate_fixed_errors()
    test.load_data(test.fb_master_query_author)
    test.populate_database()
    # print("\n\nDUPLICATES: " + str(test.findDupes()))

    test.load_data(test.fb_master_query_nested_source)
    test.populate_database()
    print("\n\nDUPLICATES: " + str(test.findDupes()))
    print(test.nesteds)

    # test.generate_error_txt()
    test.close()


