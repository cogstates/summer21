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
    max_length = 0

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
        raw_text_orig = f.read()
        tree = xmltodict.parse(raw_text_orig)['DOC']
        # self.pp.pprint(tree)
        # text = tree['TEXT']['P']
        # full_text = ' '.join(text)

        # populating dictionary of paragraph start indicies "<P>"
        paragraph_indices = {}
        raw_text = raw_text_orig
        p_start = raw_text.find('<P>\n')

        while p_start != -1:

            p_end = raw_text.find('\n</P>', p_start)
            paragraph_indices[p_start] = p_end
            p_start = raw_text.find('<P>\n', p_start + len('<P>\n'))

        p_starts = sorted(paragraph_indices.keys())
        tokenized_sentences = []
        i = 1

        text = []
        for p_start in p_starts:
            p_end = paragraph_indices[p_start]
            paragraph = raw_text_orig[p_start + len('<P>\n'):p_end]
            text.append(paragraph)

        for paragraph in text:

            sent_tokenized_paragraph = sent_tokenize(paragraph)


            for s in sent_tokenized_paragraph:
                final_paragraph = []

                sent_offset_start = raw_text_orig.index(s)
                sent_offset_end = sent_offset_start + len(s)
                final_paragraph.append(i)
                final_paragraph.append(sent_offset_start)
                final_paragraph.append(sent_offset_end)
                final_paragraph.append(s)
                assert s == raw_text_orig[sent_offset_start:sent_offset_end]
                i += 1

                tokenized_sentences.append(final_paragraph)

        entry = {'id': tree['@id'], 'headline': tree['HEADLINE'],
                 'raw_text': raw_text_orig, 'split_text': text,
                 'sentences': tokenized_sentences,
                 'paragraph_indices': paragraph_indices}
        self.source_text[tree['@id']] = entry

    # loading ERE data
    def load_ere(self):
        pass

    def parse_best_file(self, f_name):
        f = open(f_name, encoding='utf-8')
        tree = xmltodict.parse(f.read())['belief_sentiment_doc']
        source_id = f_name[f_name.index('annotation/') + len('annotation/'):f_name.index('.best.xml')]

        source_text = self.source_text[source_id]
        raw_source_text = source_text['raw_text']

        relation_belief_annotations = tree['belief_annotations']['relations']['relation']
        event_belief_annotations = tree['belief_annotations']['events']['event']

        final_relation_belief_annotations = []
        # the structure of this list is as follows:
        # list of lists containing one element per relation belief annotation
        # inside each element (a list), we have,
        # the relative mention ID at index 0,
        # a series of 1-3 lists, each representing an atomic belief
        # inside each belief list we have label, polarity, sarcasm,
        # and source ere id, source offset, source length and source text (these are null if not present)
        # then, after these 1-3 belief lists, we have one more list representing the trigger (offset, length, text);
        # these are also null if there is no trigger

        for relation_belief_annotation in relation_belief_annotations:

            beliefs = []
            result = []

            relm_ere_id = relation_belief_annotation['@ere_id']
            belief = relation_belief_annotation['beliefs']['belief']

            result.append(relm_ere_id)

            if type(belief) != list:
                beliefs.append(belief)
            else:
                beliefs = belief

            for b in beliefs:

                this_belief = []
                label = b['@type']

                if label != 'na':
                    polarity = b['@polarity']
                    if b['@sarcasm'] == 'no':
                        sarcasm = False
                    else:
                        sarcasm = True
                else:
                    polarity, sarcasm = None, None

                this_belief.append(label)
                this_belief.append(polarity)
                this_belief.append(sarcasm)

                if 'source' in b:
                    source = b['source']
                    source_ere_id = source['@ere_id']
                    source_offset = int(source['@offset'])
                    source_length = int(source['@length'])
                    source_text = source['#text']
                    sentence_id = self.find_containing_sentence(source_id, source_offset, source_text, source_length)

                    assert raw_source_text[source_offset:source_offset + source_length] == source_text
                else:
                    source_ere_id, source_offset, source_length, source_text, sentence_id = None, None, None, None, None

                this_belief.append(source_ere_id)
                this_belief.append(source_offset)
                this_belief.append(source_length)
                this_belief.append(source_text)
                this_belief.append(sentence_id)

                result.append(this_belief)

            if 'trigger' in relation_belief_annotation:
                trigger = relation_belief_annotation['trigger']
                trigger_offset = int(trigger['@offset'])
                trigger_length = int(trigger['@length'])
                trigger_text = trigger['#text']
                sentence_id = self.find_containing_sentence(source_id, trigger_offset, trigger_text, trigger_length)

                assert raw_source_text[trigger_offset:trigger_offset+trigger_length] == trigger_text
            else:
                trigger_offset, trigger_length, trigger_text, sentence_id = None, None, None, None

            result.append([trigger_offset, trigger_length, trigger_text, sentence_id])

            final_relation_belief_annotations.append(result)

        self.best[source_id] = {'relation_belief_annotations': final_relation_belief_annotations}

    def find_containing_sentence(self, source_id, offset, text, length):
        for sentence in self.source_text[source_id]['sentences']:
            sentence_id = sentence[0]
            sentence_offset_start, sentence_offset_end = sentence[1], sentence[2]
            sentence_text = sentence[3]
            if sentence_offset_start <= offset <= sentence_offset_end:
                assert text in sentence_text
                return sentence_id
        return None







if __name__ == "__main__":
    test = BEST2MASTER()
    test.load_data()
    source_text_ids = [key for key in test.source_text.keys()]
    # test.pp.pprint(test.source_text[source_text_ids[0]])
    print(source_text_ids[1])
    # test.pp.pprint((test.source_text[source_text_ids[1]]['split_text']))
    # test.pp.pprint((test.source_text[source_text_ids[1]]['sentences']))
    test.pp.pprint((test.best[source_text_ids[1]]['relation_belief_annotations']))
    # test.pp.pprint((test.best[source_text_ids[1]]['relation_belief_annotations']))
    # print(test.best[source_text_ids[0]]['max_offset'])
    # print(len(test.source_text[source_text_ids[0]]['full_text']) + len(test.source_text[source_text_ids[0]]['headline']))
