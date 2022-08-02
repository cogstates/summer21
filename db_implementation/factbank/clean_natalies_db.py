import sqlite3

dirty_con = sqlite3.connect('factbank_data.db')
dirty_cur = dirty_con.cursor()

clean_con = sqlite3.connect('factbank_data_CLEAN.db')
clean_cur = clean_con.cursor()



