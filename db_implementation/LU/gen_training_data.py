import pandas as pd
import sqlite3

# ...sentence...####(head text, label as number)
# 0 CB, 1 NCB, 2 NA
def gen_LU_data_classic():
    sql_data = cur.execute('''SELECT * FROM
             (SELECT s.file, s.file_sentence_id,
       s.sentence, m.token_text target_head, m.token_offset_start target_offset_start,
       m.token_offset_end target_offset_end, a.label
FROM attitudes a
    JOIN mentions m on m.token_id = a.target_token_id
    JOIN sentences s on m.sentence_id = s.sentence_id
    JOIN sources s2 on s2.source_id = a.source_id)
    
    ORDER BY file, file_sentence_id;
    
    ''').fetchall()

    data = []
    encountered_sentences = {}

    for row in sql_data:
        file = row[0]
        sentence = row[2]
        target_head = row[3]
        target_head_offset_start = row[4]
        target_head_offset_end = row[5]
        label = {'CB': 'true', 'NCB': 'false', 'NA': 'unknown'}[row[6]]

        entry = f'({target_head}, {label})'

        key = (file, sentence)

        if key not in encountered_sentences:
            encountered_sentences[key] = [entry]
        else:
            current_list = encountered_sentences[key]
            current_list.append(entry)
            encountered_sentences[key] = current_list


    for key, value in encountered_sentences.items():
        file, sentence = key
        tuples = value

        target_data = ''
        for t in tuples:
            target_data += f'{t}; '
        target_data = target_data[:-2]
        formatted_row = [f'{file}', f'LU Factuality: {sentence}', f'{target_data}']
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

    # print(train)
    # print(test)
    # print(dev)
    #
    # print(boundary, total_size - boundary)
    # print(before_boundary_file, after_boundary_file)

    return train, test, dev




if __name__ == "__main__":
    con = sqlite3.connect("LU_master.db")
    cur = con.cursor()

    train, test, dev = gen_LU_data_classic()
    for df in [(train, 'train'), (test, 'test'), (dev, 'dev')]:
        df[0].loc[:, 'input_text':'target_text'].to_csv(f'formatted_data/LU_initial/{df[1]}.csv')

    con.close()