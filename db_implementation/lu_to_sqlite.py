import sqlite3
con = sqlite3.connect('lu_data.db')
cur = con.cursor()

##### CREATES THE EMPTY DATABASE TABLES FOR ALL OF THE FACTBANK TXT FILES #####

# self.doc_id = this_doc_id
# self.sentence_id = this_sentence_id
# self.text = this_text
# self.head_start = this_head_start
# self.head_end = this_head_end
# self.belief = this_belief
# self.head = this_head

# fb_corefSource.txt
cur.execute('''CREATE TABLE IF NOT EXISTS lu_data
               (doc_id INTEGER,
               sentence_id INTEGER,
               text TEXT,
               head_start INTEGER,
               head_end INTEGER,
               belief TEXT,
               head INTEGER,
               sourceText_1 TEXT)''')

##### INSERT FACTBANK DATA INTO CREATED SQLite TABLES #####
# fb_corefSource.txt
# formatted_fb_corefSource_data = remove_pipes("./factbank_corpora/fb_corefSource.txt")
# for entry in formatted_fb_corefSource_data:
#   cur.execute("INSERT INTO fb_corefSource VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(entry))


con.commit()
con.close()