import sqlite3
con = sqlite3.connect('factbank_data.db')
cur = con.cursor()

def remove_pipes(fpath):
  formattedList = []
  with open(fpath) as f:
    for line in f:
      formattedList.append(line.split("|||"))
  return formattedList

##### CREATES THE EMPTY DATABASE TABLES FOR ALL OF THE FACTBANK TXT FILES #####

# fb_corefSource.txt
cur.execute('''CREATE TABLE IF NOT EXISTS fb_corefSource
               (file TEXT,
               sentId INTEGER,
               corefSourceId TEXT,
               sourceId_1 TEXT,
               sourceId_2 TEXT,
               sourceLoc_1 INTEGER,
               sourceLoc_2 INTEGER,
               sourceText_1 TEXT,
               sourceText_2 TEXT)''')

#fb_event.txt
cur.execute('''CREATE TABLE IF NOT EXISTS fb_event
               (file TEXT,
               sentId INTEGER,
               eId TEXT,
               eiId TEXT,
               eText TEXT)''')

#fb_factValue
cur.execute('''CREATE TABLE IF NOT EXISTS fb_factValue
               (file TEXT,
               sentId INTEGER,
               fvId TEXT,
               eId TEXT,
               eiId TEXT,
               relSourceId TEXT,
               eText TEXT,
               relSourceText TEXT,
               factValue TEXT)''')

#fb_relSource.txt
cur.execute('''CREATE TABLE IF NOT EXISTS fb_relSource
               (file TEXT,
               sentId INTEGER,
               relSourceId TEXT,
               relSourceText TEXT)''')

#fb_sip.txt
cur.execute('''CREATE TABLE IF NOT EXISTS fb_sip
               (file TEXT,
               sentId INTEGER,
               sip_eId TEXT,
               sip_eiId TEXT,
               sip_Text TEXT)''')

#fb_sipAndSource.txt
cur.execute('''CREATE TABLE IF NOT EXISTS fb_sipAndSource
               (file TEXT,
               sentId INTEGER,
               sip_eId TEXT,
               eip_eiId TEXT,
               sip_Text TEXT,
               sourceId TEXT,
               sourceText TEXT)''')

#fb_source.txt
cur.execute('''CREATE TABLE IF NOT EXISTS fb_source
               (file TEXT,
               sentId INTEGER,
               sourceId TEXT,
               sourceLoc INTEGER,
               sourceText TEXT)''')

#fb_sourceString.txt
cur.execute('''CREATE TABLE IF NOT EXISTS fb_sourceString
               (file TEXT,
               sentId INTEGER,
               sourceId TEXT,
               sourceLoc INTEGER,
               sourceText TEXT)''')

#files.txt
cur.execute('''CREATE TABLE IF NOT EXISTS files
               (file TEXT,
               corpus TEXT)''')

#offsets.txt
cur.execute('''CREATE TABLE IF NOT EXISTS offsets
               (file TEXT,
               offsetInit INTEGER,
               offsetEnd INTEGER,
               sentId INTEGER,
               tokLoc INTEGER,
               text TEXT)''')

#sentences.txt
cur.execute('''CREATE TABLE IF NOT EXISTS sentences
               (file TEXT,
               sentId INTEGER,
               sent TEXT)''')

#tml_alink.txt
cur.execute('''CREATE TABLE IF NOT EXISTS tml_alink
               (file TEXT,
               lId TEXT,
               eId_1 TEXT,
               eId_2 TEXT,
               eiId_1 TEXT,
               eiId_2 TEXT,
               relType TEXT,
               signalId TEXT,
               eText_1 TEXT,
               eText_2 TEXT,
               signalText TEXT)''')

#tml_event.txt
cur.execute('''CREATE TABLE IF NOT EXISTS tml_event
               (file TEXT,
               sentId INTEGER,
               eId TEXT,
               eClass TEXT,
               eText TEXT)''')

#tml_instance.txt
cur.execute('''CREATE TABLE IF NOT EXISTS tml_instance
               (file TEXT,
               eId TEXT,
               eiId TEXT,
               tense TEXT,
               aspect TEXT,
               pos TEXT,
               polarity TEXT,
               modality TEXT,
               cardinality TEXT,
               signalId TEXT)''')

#tml_signal.txt
cur.execute('''CREATE TABLE IF NOT EXISTS tml_signal
               (file TEXT,
               sentId INTEGER,
               signalId TEXT,
               signalText TEXT,
               tokenText TEXT)''')

#tml_slink.txt
cur.execute('''CREATE TABLE IF NOT EXISTS tml_slink
               (file TEXT,
               lId TEXT,
               eId_1 TEXT,
               eId_2 TEXT,
               eiId_1 TEXT,
               eiId_2 TEXT,
               relType TEXT,
               signalId TEXT,
               eText_1 TEXT,
               eText_2 TEXT,
               signalText TEXT)''')

#tml_timex3.txt
## temporalFunction holds a boolean value, stored as string for now, happy to convert to 0/1 F/T system
cur.execute('''CREATE TABLE IF NOT EXISTS tml_timex3
               (file TEXT,
               sentId INTEGER,
               tId TEXT,
               type TEXT,
               value TEXT,
               mod TEXT,
               functionInDoc TEXT,
               temporalFunction TEXT,
               anchorTimeId TEXT,
               beginPoint TEXT,
               endPoint TEXT,
               freq TEXT,
               quant TEXT,
               timex TEXT,
               tokenToken TEXT)''')

#tml_tlink.txt
cur.execute('''CREATE TABLE IF NOT EXISTS tml_tlink
               (file TEXT,
               lId TEXT,
               eId_1 TEXT,
               eId_2 TEXT,
               eiId_1 TEXT,
               eiId_2 TEXT,
               tId_1 TEXT,
               tId_2 TEXT,
               relType TEXT,
               signalId TEXT,
               eText_1 TEXT,
               eText_2 TEXT,
               signalText TEXT)''')

#tokens_ling.txt
cur.execute('''CREATE TABLE IF NOT EXISTS tokens_ling
               (file TEXT,
               sentId INTEGER,
               tokLoc INTEGER,
               text TEXT,
               pos TEXT)''')

#tokens_tml.txt
cur.execute('''CREATE TABLE IF NOT EXISTS tokens_tml
               (file TEXT,
               sentId INTEGER,
               tokLoc INTEGER,
               text TEXT,
               tmlTag TEXT,
               tmlTagId TEXT,
               tmlTagLoc TEXT)''')

##### INSERT FACTBANK DATA INTO CREATED SQLite TABLES #####
# fb_corefSource.txt
formatted_fb_corefSource_data = remove_pipes("./factbank_corpora/fb_corefSource.txt")
for entry in formatted_fb_corefSource_data:
  cur.execute("INSERT INTO fb_corefSource VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(entry))

#fb_event.txt
formatted_fb_event_data = remove_pipes("./factbank_corpora/fb_event.txt")
for entry in formatted_fb_event_data:
  cur.execute("INSERT INTO fb_event VALUES (?, ?, ?, ?, ?)", tuple(entry))

#fb_factValue
formatted_fb_factValue_data = remove_pipes("./factbank_corpora/fb_factValue.txt")
for entry in formatted_fb_factValue_data:
  cur.execute("INSERT INTO fb_factValue VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(entry))

#fb_relSource.txt
formatted_fb_relSource_data = remove_pipes("./factbank_corpora/fb_relSource.txt")
for entry in formatted_fb_relSource_data:
  cur.execute("INSERT INTO fb_relSource VALUES (?, ?, ?, ?)", tuple(entry))

#fb_sip.txt
formatted_fb_sip_data = remove_pipes("./factbank_corpora/fb_sip.txt")
for entry in formatted_fb_sip_data:
  cur.execute("INSERT INTO fb_sip VALUES (?, ?, ?, ?, ?)", tuple(entry))

#fb_sipAndSource.txt
formatted_fb_sipAndSource_data = remove_pipes("./factbank_corpora/fb_sipAndSource.txt")
for entry in formatted_fb_sipAndSource_data:
  cur.execute("INSERT INTO fb_sipAndSource VALUES (?, ?, ?, ?, ?, ?, ?)", tuple(entry))

#fb_source.txt
formatted_fb_source_data = remove_pipes("./factbank_corpora/fb_source.txt")
for entry in formatted_fb_source_data:
  cur.execute("INSERT INTO fb_source VALUES (?, ?, ?, ?, ?)", tuple(entry))

#fb_sourceString.txt
formatted_fb_sourceString_data = remove_pipes("./factbank_corpora/fb_sourceString.txt")
for entry in formatted_fb_sourceString_data:
  cur.execute("INSERT INTO fb_sourceString VALUES (?, ?, ?, ?, ?)", tuple(entry))

#files.txt
formatted_files_data = remove_pipes("./factbank_corpora/files.txt")
for entry in formatted_files_data:
  cur.execute("INSERT INTO files VALUES (?, ?)", tuple(entry))

#offsets.txt
formatted_offsets_data = remove_pipes("./factbank_corpora/offsets.txt")
for entry in formatted_offsets_data:
  cur.execute("INSERT INTO offsets VALUES (?, ?, ?, ?, ?, ?)", tuple(entry))

#sentences.txt
formatted_sentences_data = remove_pipes("./factbank_corpora/sentences.txt")
for entry in formatted_sentences_data:
  cur.execute("INSERT INTO sentences VALUES (?, ?, ?)", tuple(entry))

#tml_alink.txt
formatted_tml_alink_data = remove_pipes("./factbank_corpora/tml_alink.txt")
for entry in formatted_tml_alink_data:
  cur.execute("INSERT INTO tml_alink VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(entry))

#tml_event.txt
formatted_tml_event_data = remove_pipes("./factbank_corpora/tml_event.txt")
for entry in formatted_tml_event_data:
  cur.execute("INSERT INTO tml_event VALUES (?, ?, ?, ?, ?)", tuple(entry))

#tml_instance.txt
formatted_tml_instance_data = remove_pipes("./factbank_corpora/tml_instance.txt")
for entry in formatted_tml_instance_data:
  cur.execute("INSERT INTO tml_instance VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(entry))

#tml_signal.txt
formatted_tml_signal_data = remove_pipes("./factbank_corpora/tml_signal.txt")
for entry in formatted_tml_signal_data:
  cur.execute("INSERT INTO tml_signal VALUES (?, ?, ?, ?, ?)", tuple(entry))

#tml_slink.txt
formatted_tml_slink_data = remove_pipes("./factbank_corpora/tml_slink.txt")
for entry in formatted_tml_slink_data:
  cur.execute("INSERT INTO tml_slink VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(entry))

#tml_timex3.txt
formatted_tml_timex3_data = remove_pipes("./factbank_corpora/tml_timex3.txt")
for entry in formatted_tml_timex3_data:
  cur.execute("INSERT INTO tml_timex3 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(entry))

#tml_tlink.txt
formatted_tml_tlink_data = remove_pipes("./factbank_corpora/tml_tlink.txt")
for entry in formatted_tml_tlink_data:
  cur.execute("INSERT INTO tml_tlink VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(entry))

#tokens_ling.txt
formatted_tokens_ling_data = remove_pipes("./factbank_corpora/tokens_ling.txt")
for entry in formatted_tokens_ling_data:
  cur.execute("INSERT INTO tokens_ling VALUES (?, ?, ?, ?, ?)", tuple(entry))

#tokens_tml.txt
formatted_tokens_tml_data = remove_pipes("./factbank_corpora/tokens_tml.txt")
for entry in formatted_tokens_tml_data:
  cur.execute("INSERT INTO tokens_tml VALUES (?, ?, ?, ?, ?, ?, ?)", tuple(entry))

con.commit()
con.close()