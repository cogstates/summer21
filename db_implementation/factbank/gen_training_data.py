import pandas as pd
import sqlite3

# ...sentence...####(head text, label as number)
# 0 CB, 1 NCB, 2 NA

def label_norm(label_txt):
    if label_txt in ['CTu', 'NA', 'other', 'ROB']:
        return None
    return {'CT+': 'true', 'CT-': 'false',
            'PS+': 'possibly true', 'PR+': 'possibly true',
            'PS-': 'possibly false', 'PR-': 'possibly false',
            'Uu': 'unknown'}[label_txt]
def gen_fb_data_author_only():
    sql_data = cur.execute('''SELECT target_data.file, target_data.file_sentence_id, target_data.sentence, target_data.target_head, 
    target_data.label, target_data.target_offset_start, target_data.target_offset_end, source_data.source_text FROM
             (SELECT a.attitude_id, s.sentence_id, s.sentence, s.file, s.file_sentence_id, m.token_text target_head,
       m.token_offset_start target_offset_start, m.token_offset_end target_offset_end,
       m.phrase_offset_start target_span_start, m.phrase_offset_end target_span_end,
       m.phrase_text target_span, m.token_id target_token, a.label
FROM attitudes a
    JOIN mentions m on m.token_id = a.target_token_id
    JOIN sentences s on m.sentence_id = s.sentence_id) target_data
JOIN (SELECT a.attitude_id, m.token_text source_text,
       s.nesting_level, m.token_offset_start source_offset_start, m.token_offset_end source_offset_end,
       m.phrase_offset_start source_span_start, m.phrase_offset_end source_span_end,
       m.phrase_text source_span,
       m.token_id source_token_id, s.parent_source_id, s.source_id
FROM attitudes a
    JOIN sources s on a.source_id = s.source_id
    JOIN mentions m on s.token_id = m.token_id
    JOIN sentences s2 on m.sentence_id = s2.sentence_id) source_data on target_data.attitude_id = source_data.attitude_id
where source_text in ('AUTHOR')
order by file, file_sentence_id;''').fetchall()

    data = []
    encountered_sentences = {}
    for row in sql_data:
        file = row[0]
        file_sentence_id = row[1]
        sentence = row[2]
        target_head = row[3]
        label = label_norm(row[4])
        target_offset_start = row[5]
        target_offset_end = row[6]

        if label is None:
            continue

        sentence_key = (file, file_sentence_id)
        entry = (target_head, target_offset_start, target_offset_end, label)

        if sentence_key not in encountered_sentences:
            encountered_sentences[sentence_key] = [sentence, [entry]]
        else:
            current_list = encountered_sentences[sentence_key][1]
            current_list.append(entry)
            encountered_sentences[sentence_key] = \
                [sentence, current_list]

    for key, value in encountered_sentences.items():
        file, file_sentence_id = key
        sentence, tuples = value

        gold_sentence = ''
        start_here = 0

        for t in tuples:
            target_head = t[0]
            target_offset_start, target_offset_end = t[1], t[2]
            label = t[3]

            gold_sentence += f'{sentence[start_here:target_offset_start]}[{target_head}|{label}]'
            start_here = target_offset_end

        gold_sentence += sentence[start_here:]

        formatted_row = [f'{file}', f'source target factuality: {sentence}', f'{gold_sentence}']
        data.append(formatted_row)

    return get_final_split(data)


def get_final_split(data):
    df = pd.DataFrame(data)
    df.columns = ['file', 'input_text', 'target_text']

    total_size = len(df.index)
    boundary = int(0.8 * total_size)
    train = df.iloc[0:boundary]
    test = df.iloc[boundary + 1:]

    before_boundary_file = train.iat[-1, 0]
    after_boundary_file = test.iat[0, 0]
    if before_boundary_file == after_boundary_file:
        spillover = train[train['file'] == before_boundary_file]
        test = pd.concat([test, spillover])
        train.drop(spillover.index, inplace=True)

    dev = test.iloc[len(test.index) // 2:]
    test.drop(dev.index, inplace=True)

    return train, test, dev


if __name__ == "__main__":
    con = sqlite3.connect("fb_master.db")
    cur = con.cursor()

    train, test, dev = gen_fb_data_author_only()
    for df in [(train, 'train'), (test, 'test'), (dev, 'dev')]:
        df[0].loc[:, 'input_text':'target_text'].to_csv(f'formatted_data/FB_initial/{df[1]}.csv', sep='|')

    con.close()
