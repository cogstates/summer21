import sqlite3
import csv


def calc_nesting_level(source_text):
    if '=' in source_text:
        return source_text[0:source_text.index('=')]
    if source_text == 'AUTHOR':
        return 'AUTHOR'
    return source_text[:source_text.index('_')]


con = sqlite3.connect('factbank_data_CLEAN.db')
cur = con.cursor()

sql_return = cur.execute("""
SELECT distinct s.file, s.sentId, s.sent, t.tokLoc, f.eText, f.relSourceText, o.offsetInit, o.offsetEnd, o.sentId
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
               AND f.eText = t.text;""")
headers = ['filename', 'sentId', 'sent', 'eText', 'relSourceText']
with open('natalie_clean_v2.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerow(headers)
    for row in sql_return:
        row = list(row)
        file = row[0]
        sent_id = row[1]
        sent = row[2]
        etext = row[4].replace('\\', '')
        source_text = calc_nesting_level(row[5]).replace('\\', '')
        write.writerow([file, sent_id, sent, etext, source_text])



