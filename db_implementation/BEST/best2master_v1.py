# author: tyler osborne
# osbornty@bc.edu
# 09/18/2022

import xmltodict
import os
import pprint
from ddl import DDL
from nltk.tokenize import sent_tokenize

# TODO:
# for message boards, make it so that quotes are stored when the text tag's text attribute is empty.
# also, save author metadata.
# to ask in meeting: do we care about _____ (Look at ERE and BEST metadata)


class BEST2MASTER:
    pp = pprint.PrettyPrinter()

    def __init__(self):
        self.source_text = {}
        self.ere = {}
        self.best = {}
        self.load_source_text()

    # loading all source text

    # source text will be stored as a dictionary of dictionaries,
    # each inner dictionary representing one source XML file,
    # accessed by its document ID. each entry inside a given inner dictionary
    # will represent a discrete chunk of text as it is split up in the XML files.
    # keys for these entries will either be taken directly from the associated XML entry,
    # otherwise we will auto-generate them as ascending integers.
    # headlines will always have an ID of 0 in cases where the XML does not give us IDs
    def load_source_text(self):
        directory = '../../raw_corpora/BEST/english/data/source/'
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            ext = os.path.splitext(f)[1]
            if ext == '.xml' and 'ENG_DF' not in f:
                self.parse_source_file(f)

    # parsing a single source XML file
    def parse_source_file(self, f_name):
        f = open(f_name, encoding='utf-8')
        tree = xmltodict.parse(f.read())['DOC']
        # self.pp.pprint(tree)
        text = tree['TEXT']['P']
        full_text = ' '.join(text)

        tokenized_sentences = []
        for sen in text:
            output = sent_tokenize(sen)
            for s in output:
                tokenized_sentences.append(s)

        entry = {'id': tree['@id'], 'headline': tree['HEADLINE'],
                 'full_text': full_text, 'sentences': tokenized_sentences}
        self.pp.pprint(entry)
        print('\n\n\n')

    # loading ERE data
    def load_ere(self):
        pass

    # loading BEST annotation data
    def load_best(self):
        pass


if __name__ == "__main__":
    test = BEST2MASTER()
    test.load_source_text()
