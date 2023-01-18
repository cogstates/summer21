import pandas as pd
import sqlite3

# ...sentence...####(head text, label as number)
# 0 CB, 1 NCB, 2 NA
def gen_FB_data_classic():
    sql_data = cur.execute('''SELECT distinct target_data.file, target_data.file_sentence_id, target_data.sentence, target_data.target_head, 
    target_data.label, source_data.source_text FROM
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
where source_text not in ('AUTHOR', 'GEN', 'DUMMY')
order by file, file_sentence_id;''').fetchall()

    data = []
    encountered_sentences = {}
    for row in sql_data:
        file = row[0]
        file_sentence_id = row[1]
        sentence = row[2]
        target_head = row[3]

        if row[4] in ['CTu', 'NA', 'other']:
            continue

        label = {'CT+': 1, 'PR+': 2, 'PS+': 3, 'CT-': 4, 'PR-': 5, 'PS-': 6, 'ROB': 8, 'Uu': 9}[row[4]]
        source = row[5]
        sentence_key = (file, file_sentence_id)
        if sentence_key not in encountered_sentences:
            encountered_sentences[sentence_key] = [sentence, f'(source = {source}, target = {target_head}, {label});']
        else:
            encountered_sentences[sentence_key] = \
                [sentence, encountered_sentences[sentence_key][1] +
                 f' (source = {source}, target = {target_head}, {label});']

    for key, value in encountered_sentences.items():
        file, file_sentence_id = key
        sentence, target_data = value
        formatted_row = [f'{file}', f'source target factuality: {sentence}', f'{target_data}']
        data.append(formatted_row)

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

    train, test, dev = gen_FB_data_classic()
    for df in [(train, 'train'), (test, 'test'), (dev, 'dev')]:
        df[0].loc[:, 'input_text':'target_text'].to_csv(f'formatted_data/FB_initial/{df[1]}.csv', sep='|')

    con.close()