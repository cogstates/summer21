# author: tyler osborne
# osbornty@bc.edu
# 04/03/2021

import sqlite3
from ddl import DDL
from progress.bar import Bar
from os.path import exists
from os import remove
from time import sleep


class FB2Master:
    # static constants to index into the raw factbank dataset
    FILE = 0
    SENTENCE_ID = 1
    SENTENCE = 2
    RAW_OFFSET_INIT = 2
    REL_SOURCE_TEXT = 2

    def __init__(self):
        self.fb_con = sqlite3.connect("factbank_data.db")
        self.fb_cur = self.fb_con.cursor()

        self.create_tables()
        self.ma_con = sqlite3.connect("fb_master.db")
        self.ma_cur = self.ma_con.cursor()

        self.fb_sentences_query = """
        SELECT DISTINCT s.file, s.sentid, s.sent
        FROM sentences s
        ORDER BY s.file, s.sentid;"""

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
                       AND f.eText = t.text;
                       --AND f.relSourceText = "'AUTHOR'";"""
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
        self.rel_source_texts = {}
        self.source_offsets = {}
        self.eid_errors = []

    def create_tables(self):
        db = DDL('fb')
        db.create_tables()
        db.close()

    def load_rel_source_texts(self):
        rel_source_data = self.fb_cur.execute('SELECT file, sentId, relSourceText FROM fb_relSource;').fetchall()
        for row in rel_source_data:
            key = (row[self.FILE], row[self.SENTENCE_ID])
            value = str(row[self.REL_SOURCE_TEXT])[1:-2]
            if key not in self.rel_source_texts:
                self.rel_source_texts[key] = [value]
            else:
                self.rel_source_texts[key].append(value)

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

    def load_data(self):
        # building dictionary of initial offsets for later calculation of sentence-based token offsets
        offsets_sql_return = self.fb_cur.execute(self.offsets_query)
        for row in offsets_sql_return:
            self.initial_offsets[(row[self.FILE], row[self.SENTENCE_ID])] = row[self.RAW_OFFSET_INIT]

        # inserting sentences
        sentences_sql_return = self.fb_cur.execute(self.fb_sentences_query).fetchall()
        bar = Bar('Sentences Processed', max=len(sentences_sql_return))

        for row in sentences_sql_return:
            if row[self.SENTENCE_ID] == 0:
                continue
            row = list(row)

            # removing enclosing single quotes where necessary
            row[self.SENTENCE] = str(row[self.SENTENCE][1:-2].replace("\\", "").replace("`", ""))

            # inserting sentences to our sentences table
            self.ma_cur.execute('INSERT INTO sentences (file, file_sentence_id, sentence)'
                                'VALUES (?, ?, ?);',
                                (row[self.FILE], row[self.SENTENCE_ID], row[self.SENTENCE]))

            # getting back whatever unique sentence id was just generated by the above insert
            global_sentence_id = self.ma_cur.execute('SELECT sentence_id FROM sentences WHERE sentence = ?',
                                                     (row[self.SENTENCE],)).fetchone()[0]

            rel_source_key = (row[self.FILE], row[self.SENTENCE_ID])
            if rel_source_key not in self.rel_source_texts:
                self.rel_source_texts[rel_source_key] = ['AUTHOR']
            sources = self.rel_source_texts[rel_source_key]

            # for each nesting level 0 --> 3, get all sources from fb_relsource
            for current_nesting_level in range(0, 3):
                for rel_source_text in sources:
                    nesting_level, relevant_source = self.calc_nesting_level(rel_source_text)

                    if nesting_level != current_nesting_level:
                        continue

                    # getting the source offsets
                    source_offsets_key = (row[self.FILE], row[self.SENTENCE_ID])
                    if source_offsets_key not in self.source_offsets:
                        self.source_offsets[source_offsets_key] = (None, None, relevant_source)
                    source_offsets = self.source_offsets[source_offsets_key]

                    offset_start, offset_end, success = self.calc_offsets(row[self.FILE], row[self.SENTENCE_ID],
                                                                          row[self.SENTENCE],
                                                                          source_offsets[0],
                                                                          source_offsets[1],
                                                                          relevant_source, rel_source_text)

                    # inserting on mentions
                    self.ma_cur.execute('INSERT INTO mentions (sentence_id, token_text, token_offset_start, '
                                        'token_offset_end) VALUES (?, ?, ?, ?);',
                                        (global_sentence_id, relevant_source, offset_start, offset_end))

                    # getting global token id from row we just inserted on mentions above
                    global_source_token_id = \
                        self.ma_cur.execute('SELECT token_id FROM mentions '
                                            'WHERE sentence_id = ? AND token_text = ? '
                                            'AND token_offset_start = ? AND token_offset_end = ?;',
                                            (global_sentence_id, relevant_source,
                                             offset_start, offset_end)).fetchone()[0]

                    if nesting_level == 0:
                        parent_source_id = -1
                    else:
                        # print(rel_source_text)
                        parent_source_text = self.calc_parent_source(rel_source_text)
                        parent_source_id_set = self.ma_cur.execute('SELECT source_id FROM sources '
                                                                   'WHERE sentence_id = ? '
                                                                   'AND nesting_level = ? '
                                                                   'AND [source] = ?;',
                                                                   (global_sentence_id,
                                                                    current_nesting_level - 1,
                                                                    parent_source_text))
                        parent_source_id = parent_source_id_set.fetchone()
                        parent_source_id = parent_source_id[0]

                    self.ma_cur.execute(
                        'INSERT INTO sources (sentence_id, token_id, parent_source_id, nesting_level, [source]) '
                        'VALUES (?, ?, ?, ?, ?);',
                        (global_sentence_id, global_source_token_id, parent_source_id,
                         current_nesting_level, relevant_source))

                    # getting back that source id that we just inserted
                    # print('global_sentence_id:', global_sentence_id,
                    #       'global_source_token_id:', global_source_token_id,
                    #       'parent_source_id:', parent_source_id,
                    #       'current_nesting_level:', current_nesting_level,
                    #       'relevant_source:', relevant_source)

                    attitude_source_id = self.ma_cur.execute('SELECT source_id FROM sources '
                                                             'WHERE sentence_id = ? AND token_id = ? '
                                                             'AND parent_source_id = ? AND nesting_level = ? '
                                                             'AND [source] = ?;',
                                                             (global_sentence_id, global_source_token_id,
                                                              parent_source_id, current_nesting_level,
                                                              relevant_source)).fetchone()[0]

                    eid_label_sql_return = self.fb_cur.execute('SELECT eId, factValue FROM fb_factValue '
                                                               'WHERE file = ? AND sentId = ? '
                                                               'AND relSourceText = ?',
                                                               (row[self.FILE], row[self.SENTENCE_ID],
                                                                "'{}'".format(rel_source_text))).fetchone()
                    if eid_label_sql_return is None:
                        self.eid_errors.append((row[self.FILE], row[self.SENTENCE_ID], rel_source_text))
                        continue

                    eid = eid_label_sql_return[0]
                    fact_value = eid_label_sql_return[1][1:-2]

                    target_sql_return = self.fb_cur.execute('SELECT tokLoc, text FROM tokens_tml '
                                                            'WHERE file = ? AND sentId = ? '
                                                            'AND tmlTagId = ?;',
                                                            (row[self.FILE], row[self.SENTENCE_ID], eid)).fetchone()
                    tok_loc = target_sql_return[0]
                    target_head = target_sql_return[1][1:-1]

                    # getting target offsets before inserting on mentions
                    target_offsets_sql_return = self.fb_cur.execute('SELECT offsetInit, offsetEnd FROM offsets '
                                                                    'WHERE file = ? AND sentId = ? '
                                                                    'AND tokLoc = ?;',
                                                                    (row[self.FILE], row[self.SENTENCE_ID],
                                                                     tok_loc)).fetchone()
                    target_offset_start = target_offsets_sql_return[0]
                    target_offset_end = target_offsets_sql_return[1]

                    target_offset_start, target_offset_end, success = self.calc_offsets(row[self.FILE],
                                                                                        row[self.SENTENCE_ID],
                                                                                        row[self.SENTENCE],
                                                                                        target_offset_start,
                                                                                        target_offset_end,
                                                                                        target_head,
                                                                                        rel_source_text)

                    # inserting target on mentions
                    self.ma_cur.execute('INSERT INTO mentions (sentence_id, token_text, '
                                        'token_offset_start, token_offset_end) '
                                        'VALUES (?, ?, ?, ?);', (global_sentence_id, target_head,
                                                                 target_offset_start, target_offset_end))

                    # continue

                    # getting back the token id we just inserted
                    target_token_id = self.ma_cur.execute('SELECT token_id FROM mentions '
                                                          'WHERE sentence_id = ? AND token_text = ? '
                                                          'AND token_offset_start = ? AND token_offset_end = ?;',
                                                          (global_sentence_id, target_head, target_offset_start,
                                                           target_offset_end)).fetchone()[0]

                    # finally, insert on attitudes
                    self.ma_cur.execute('INSERT INTO attitudes (source_id, target_token_id, label, label_type) '
                                        'VALUES (?, ?, ?, ?)',
                                        (attitude_source_id, target_token_id, fact_value, 'Belief'))
            bar.next()
        bar.finish()

    def calc_parent_source(self, source_text):
        if source_text == 'AUTHOR':
            return None
        start_index = source_text.index('_') + 1
        parent_source = source_text[start_index:]
        if parent_source.count('_') > 0:
            parent_source = parent_source[:parent_source.index('_')]
        if '=' in parent_source:
            parent_source = parent_source[:parent_source.index('=')]
        return parent_source

    def calc_nesting_level(self, source_text):
        # print(source_text)
        nesting_level = source_text.count('_')
        if '=' in source_text:
            return nesting_level, source_text[0:source_text.index('=')]
        if source_text == 'AUTHOR':
            return 0, 'AUTHOR'
        return nesting_level, source_text[:source_text.index('_')]

    def calc_offsets(self, file, sent_id, raw_sentence, offset_start, offset_end, head, rel_source_text):
        # calculating the initial offset, since the indicies are file-based and not sentence-based in the DB

        if (offset_start is None and offset_end is None) or head is None:
            return -1, -1, True

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
                     raw_sentence, result_sentence, rel_source_text)
            if error_key not in self.errors:
                self.errors[error_key] = [entry]
            else:
                self.errors[error_key].append(entry)

        return offset_start, offset_end, success

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
    test.load_rel_source_texts()
    test.load_source_offsets()
    test.load_data()
    for i in range(5):
        print(test.eid_errors[i])
    # print("\n\nDUPLICATES: " + str(test.findDupes()))

    # test.load_data(test.fb_master_query_nested_source)
    # test.populate_database()
    # print("\n\nDUPLICATES: " + str(test.findDupes()))

    # test.generate_error_txt()
    test.close()
