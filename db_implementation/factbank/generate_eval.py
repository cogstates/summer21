import sqlite3, csv, random

class GenerateEval:

    UN_NESTED_QUERY = """SELECT distinct target_data.*, source_data.* FROM
             (SELECT a.attitude_id, s.sentence_id, s.sentence, s.file, s.file_sentence_id, m.token_text target_head,
       m.token_offset_start target_offset_start, m.token_offset_end target_offset_end,
       m.phrase_offset_start target_span_start, m.phrase_offset_end target_span_end,
       SUBSTR(s.sentence, m.phrase_offset_start, m.phrase_offset_end - m.phrase_offset_start + 1) target_span, m.token_id target_token, a.label
FROM attitudes a
    JOIN mentions m on m.token_id = a.target_token_id
    JOIN sentences s on m.sentence_id = s.sentence_id) target_data
JOIN (SELECT a.attitude_id, m.token_text source_text,
       s.nesting_level, m.token_offset_start source_offset_start, m.token_offset_end source_offset_end,
       m.phrase_offset_start source_span_start, m.phrase_offset_end source_span_end,
       SUBSTR(s2.sentence, m.phrase_offset_start, m.phrase_offset_end - m.phrase_offset_start + 1) source_span,
       m.token_id source_token_id, s.parent_source_id, s.source_id
FROM attitudes a
    JOIN sources s on a.source_id = s.source_id
    JOIN mentions m on s.token_id = m.token_id
    JOIN sentences s2 on m.sentence_id = s2.sentence_id) source_data on target_data.attitude_id = source_data.attitude_id
where nesting_level = 0"""

    NESTED_QUERY = """SELECT distinct target_data.*, source_data.* FROM
             (SELECT a.attitude_id, s.sentence_id, s.sentence, s.file, s.file_sentence_id, m.token_text target_head,
       m.token_offset_start target_offset_start, m.token_offset_end target_offset_end,
       m.phrase_offset_start target_span_start, m.phrase_offset_end target_span_end,
       SUBSTR(s.sentence, m.phrase_offset_start, m.phrase_offset_end - m.phrase_offset_start + 1) target_span, m.token_id target_token, a.label
FROM attitudes a
    JOIN mentions m on m.token_id = a.target_token_id
    JOIN sentences s on m.sentence_id = s.sentence_id) target_data
JOIN (SELECT a.attitude_id, m.token_text source_text,
       s.nesting_level, m.token_offset_start source_offset_start, m.token_offset_end source_offset_end,
       m.phrase_offset_start source_span_start, m.phrase_offset_end source_span_end,
       SUBSTR(s2.sentence, m.phrase_offset_start, m.phrase_offset_end - m.phrase_offset_start + 1) source_span,
       m.token_id source_token_id, s.parent_source_id, s.source_id
FROM attitudes a
    JOIN sources s on a.source_id = s.source_id
    JOIN mentions m on s.token_id = m.token_id
    JOIN sentences s2 on m.sentence_id = s2.sentence_id) source_data on target_data.attitude_id = source_data.attitude_id
where nesting_level > 0"""

    def __init__(self):
        self.con = sqlite3.connect("fb_master.db")
        self.cur = self.con.cursor()
        self.un_nested_data = []
        self.nested_data = []

    def close(self):
        self.con.close()

    def load_data(self):
        self.un_nested_data = self.cur.execute(self.UN_NESTED_QUERY).fetchall()
        self.nested_data = self.cur.execute(self.NESTED_QUERY).fetchall()

    def gen_csv(self, output_type):
        # header = ["sentence_id", "sentence", "file", "target_head",
        #           "target_span", "target_token", "label", "source_text", "nesting_level",
        #           "source_span",
        #           "Tyler", "Amittai", "Owen", "Ghasem"]
        header = ["attitude_id", "sentence_id", "sentence", "file", "file_sentence_id", "target_head",
                  "target_offset_start", "target_offset_end", "target_span_start", "target_span_end",
                  "target_span", "target_token", "label", "attitude_id", "source_text", "nesting_level",
                  "source_offset_start", "source_offset_end", "source_span_start", "source_span_end",
                  "source_span", "source_token_id", "parent_source_id", "source_id",
                  "Tyler", "Amittai", "Owen", "Ghasem"]

        with open('eval_{}.csv'.format(output_type), mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            if output_type == 'unnested':
                data = self.un_nested_data
            elif output_type == 'nested':
                data = self.nested_data

            data_range = range(0, len(data))

            tyler = random.sample(data_range, 13)
            amittai = random.sample(data_range, 13)
            owen = random.sample(data_range, 25)
            ghasem = random.sample(data_range, 13)

            # print(tyler)
            # print(amittai)
            # print(owen)
            # print(ghasem)

            for i in range(len(data)):
                row = list(data[i])

                if i in tyler:
                    row.extend(['X', None, None, None])
                elif i in amittai:
                    row.extend([None, 'X', None, None])
                elif i in owen:
                    row.extend([None, None, 'X', None])
                elif i in ghasem:
                    row.extend([None, None, None, 'X'])

                writer.writerow(row)


if __name__ == "__main__":
    eval = GenerateEval()
    eval.load_data()
    eval.gen_csv('nested')
    eval.gen_csv('unnested')
    eval.close()

