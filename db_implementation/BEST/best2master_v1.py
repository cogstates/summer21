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

    # loading all source text
    def load_data(self):
        self.load_xml('source')
        self.load_xml('annotation')

    # source text will be stored as a dictionary of dictionaries,
    # each inner dictionary representing one source XML file,
    # accessed by its document ID. each entry inside a given inner dictionary
    # will represent a discrete chunk of text as it is split up in the XML files.
    # keys for these entries will either be taken directly from the associated XML entry,
    # otherwise we will auto-generate them as ascending integers.
    # headlines will always have an ID of 0 in cases where the XML does not give us IDs
    def load_xml(self, folder):
        directory = '../../raw_corpora/BEST/english/data/{}/'.format(folder)
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            ext = os.path.splitext(f)[1]
            if ext == '.xml' and 'NYT_ENG' in f and 'ENG_DF' not in f:
                if folder == 'source':
                    self.parse_source_file(f)
                elif folder == 'annotation':
                    self.parse_best_file(f)
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
        self.source_text[tree['@id']] = entry
        print('\n\n\n')

    # loading ERE data
    def load_ere(self):
        pass

    def parse_best_file(self, f_name):
        f = open(f_name, encoding='utf-8')
        tree = xmltodict.parse(f.read())['belief_sentiment_doc']
        source_id = f_name[f_name.index('annotation/') + len('annotation/'):f_name.index('.best.xml')]
        # source = self.source_text[source_id]
        entry = {source_id: tree}
        belief_annotations = tree['belief_annotations']
        belief_relations = belief_annotations['relations']['relation']


        max_offset = 0
        for belief_relation in belief_relations:
            if 'trigger' in belief_relation and int(belief_relation['trigger']['@offset']) > max_offset:
                max_offset = int(belief_relation['trigger']['@offset'])

        belief_events = belief_annotations['events']['event']

        entry['belief_relations'] = belief_relations
        entry['belief_events'] = belief_events
        entry['max_offset'] = max_offset

        self.best[source_id] = entry



if __name__ == "__main__":
    test = BEST2MASTER()
    test.load_data()
    source_text_ids = [key for key in test.source_text.keys()]
    # test.pp.pprint(test.source_text[source_text_ids[0]])
    test.pp.pprint(test.best[source_text_ids[0]]['belief_relations'])
    print(test.best[source_text_ids[0]]['max_offset'])
    print(len(test.source_text[source_text_ids[0]]['full_text']) + len(test.source_text[source_text_ids[0]]['headline']))
