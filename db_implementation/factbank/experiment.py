import sqlite3
QUERY = """SELECT DISTINCT s.file, s.sent, t.tokLoc, t.text, f.factValue, o.offsetInit, o.offsetEnd, o.sentId,
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
SENTENCE = 1
REL_SOURCE_TEXT = 8


def calc_source(source_text):
    if '=' in source_text:
        return source_text[0:source_text.index('=')]
    return source_text[:source_text.index('_')]

def check_source_count_in_sentence():
    con = sqlite3.connect('factbank_data.db')
    cur = con.cursor()

    sql_return = cur.execute(QUERY)
    failure_sentences = {}
    success = 0
    failure = 0
    for row in sql_return:
        row = list(row)
        row[REL_SOURCE_TEXT] = row[REL_SOURCE_TEXT][1:]
        source = calc_source(row[REL_SOURCE_TEXT])
        sentence = row[SENTENCE]
        if sentence.count(source) == 1:
            success += 1
        else:
            failure_sentences[source] = sentence
            print(sentence, source)
            failure += 1
    con.close()
    print('SUCCESSES: {0}, FAILURES: {1}'.format(success, failure))

check_source_count_in_sentence()