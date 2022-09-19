# author: tyler osborne
# osbornty@bc.edu
# 09/18/2022

import xml.etree.ElementTree as ET
import os

# TODO:
# for message boards, make it so that quotes are stored when the text tag's text attribute is empty.
# also, save author metadata.
# to ask in meeting: do we care about _____ (Look at ERE and BEST metadata)

class BEST2MASTER:

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
            if ext == '.xml':
                self.parse_source_file(f)

    # parsing a single source XML file
    def parse_source_file(self, f):
        tree = ET.parse(f)
        root = tree.getroot()
        metadata = root.attrib
        doc_id = metadata['id']
        self.source_text[doc_id] = {}

        for child in root:
            if child.tag.upper() == 'HEADLINE':
                self.source_text[doc_id][0] = child.text
            elif child.tag.upper() == 'TEXT':
                passage_id = 1
                for passage in child:
                    self.source_text[doc_id][passage_id] = passage.text
                    passage_id += 1
            elif child.tag.upper() == 'POST':
                self.source_text[doc_id][child.attrib['id']] = child.text

        # testing

        for doc_id in self.source_text:
            doc = self.source_text[doc_id]
            print(doc_id)
            if doc_id == "ENG_DF_000261_20150321_F00000081":
                for p_id in doc:
                    print(p_id, doc[p_id])
                print('\n')

    # loading ERE data
    def load_ere(self):
        pass

    # loading BEST annotation data
    def load_best(self):
        pass


if __name__ == "__main__":
    test = BEST2MASTER()
