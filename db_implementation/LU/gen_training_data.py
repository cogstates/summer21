import pandas as pd
import sqlite3

# ...sentence...####(head text, label as number)
# 0 CB, 1 NCB, 2 NA
def gen_LU_data_classic():
    sql_data = cur.execute('''SELECT * FROM
         (SELECT s.file, s.file_sentence_id, s.sentence, m.token_text target_head, 
    m.token_offset_start target_offset_start,
    m.token_offset_end target_offset_end, a.label
FROM attitudes a
    JOIN mentions m on m.token_id = a.target_token_id
    JOIN sentences s on m.sentence_id = s.sentence_id
    JOIN sources s2 on s2.source_id = a.source_id)''').fetchall()

    data = []
    for row in sql_data:
        sentence = row[2]
        target_head = row[3]
        target_head_offset_start = row[4]
        target_head_offset_end = row[5]
        label = row[6]

        formatted_row = [f'LU Predict Factuality: {sentence}', f'({target_head}, {label})']
        data.append(formatted_row)
    return pd.DataFrame(data)


if __name__ == "__main__":
    con = sqlite3.connect("LU_master.db")
    cur = con.cursor()

    df = gen_LU_data_classic()
    df.to_csv('formatted_data/test.csv', sep='|', header=['sentence', 'annotation'])

    con.close()