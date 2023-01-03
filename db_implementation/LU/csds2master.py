# author: tyler osborne
# 24 december 2022

from ddl import DDL
import sqlite3
from xml2csds import XMLCorpusToCSDSCollection
import pprint
from progress.bar import Bar
from time import time
import spacy

# a class to port CSDS objects into the unified database
# for the Language Understanding Corpus


class CSDS2Master:
    pp = pprint.PrettyPrinter()

    # initializing python data structures to mimic database schema as well as setting up database itself
    def __init__(self):
        self.unique_sentences = {}
        self.sentences = []
        self.next_sentence_id = 1

        self.mentions = []
        self.next_mention_id = 1

        self.sources = []
        self.next_source_id = 1

        self.attitudes = []
        self.next_attitude_id = 1

        self.create_tables()
        self.ma_con = sqlite3.connect("LU_master.db")
        self.ma_cur = self.ma_con.cursor()

        self.LU = XMLCorpusToCSDSCollection(
            '2010 Language Understanding',
            'CMU')
        self.collection = self.LU.create_and_get_collection().get_all_instances()[0]

        self.nlp = spacy.load("en_core_web_lg")
        self.current_doc = None

    def generate_database(self):
        start = time()
        print('Loading CSDS objects into master schema...')
        self.process_data()
        print('\nExecuting SQL Inserts...')
        self.populate_tables()
        self.close()
        print(f'Done. \nRuntime: {round(time() - start, 2)} sec')

    def populate_tables(self):

        # inserting python data into master schema
        self.ma_con.executemany('INSERT INTO sentences (sentence_id, file, file_sentence_id, sentence) '
                                'VALUES (?, ?, ?, ?);', self.sentences)
        self.ma_con.executemany('INSERT INTO mentions '
                                '(token_id, sentence_id, token_text, token_offset_start, '
                                'token_offset_end, phrase_text, phrase_offset_start, phrase_offset_end) '
                                'VALUES (?, ?, ?, ?, ?, ?, ?, ?);', self.mentions)
        self.ma_con.executemany('INSERT INTO sources '
                                '(source_id, sentence_id, token_id, parent_source_id, nesting_level, [source]) '
                                'VALUES (?, ?, ?, ?, ?, ?);', self.sources)
        self.ma_con.executemany('INSERT INTO attitudes '
                                '(attitude_id, source_id, target_token_id, label, label_type) '
                                'VALUES (?, ?, ?, ?, ?);', self.attitudes)

    def process_data(self):
        # print(len(self.collection))
        bar = Bar('Annotations Processed', max=len(self.collection))
        for example in self.collection:

            # dealing with sentences -- each CSDS object does not necessarily contain a unique sentence
            # we use a dictionary with the following function: sentence -> (sentence_id, spacy DOC object)
            sentence = example.text

            if sentence in self.unique_sentences:
                sentence_id = self.unique_sentences[sentence][0]
                self.current_doc = self.unique_sentences[sentence][1]
            else:
                self.current_doc = self.nlp(sentence)
                self.unique_sentences[sentence] = [self.next_sentence_id, self.current_doc]
                sentence_id = self.next_sentence_id
                self.next_sentence_id += 1
                self.sentences.append([sentence_id, example.file, example.sentence_id, sentence])

            # dealing with heads (i.e., mentions)
            mention_id = self.next_mention_id
            self.next_mention_id += 1

            span_start, span_end = self.get_head_span(example.head_start, example.head_end)
            span_text = sentence[span_start:span_end]

            self.mentions.append([mention_id, sentence_id, example.head,
                                  example.head_start, example.head_end,
                                  span_text, span_start, span_end])

            # dealing with sources -- all author
            source_id = self.next_source_id
            self.next_source_id += 1
            self.sources.append([source_id, sentence_id, mention_id, None, 0, 'AUTHOR'])

            # dealing with attitudes
            attitude_id = self.next_attitude_id
            self.next_attitude_id += 1
            self.attitudes.append([attitude_id, source_id, mention_id, example.belief, 'Belief'])

            bar.next()

    @staticmethod
    def get_final_span(syntactic_head_token, fb_head_token):

        # mention subtree vs children distinction in meeting!
        syntactic_head_subtree = list(syntactic_head_token.subtree)

        relevant_tokens = []

        for token in syntactic_head_subtree:
            if token.dep_ in ['cc', 'conj'] and token.i > fb_head_token.i:
                break
            relevant_tokens.append(token)

        left_edge = relevant_tokens[0].idx
        right_edge = relevant_tokens[-1].idx + len(relevant_tokens[-1].text)

        return left_edge, right_edge

    def get_head_span(self, head_token_offset_start, head_token_offset_end):

        head_span = self.current_doc.char_span(head_token_offset_start, head_token_offset_end,
                                               alignment_mode='expand')
        head_token = head_span.root

        # when above target, eliminate CC or CONJ arcs
        # if on non-FB-target verb mid-traversal, DO take CC or CONJ arcs
        # if hit AUX, don't take CC or CONJ - don't worry for now
        if head_token.dep_ == 'ROOT':
            syntactic_head_token = head_token
        else:
            syntactic_head_token = None
            ancestors = list(head_token.ancestors)
            ancestors.insert(0, head_token)

            if len(ancestors) == 1:
                syntactic_head_token = ancestors[0]
            else:
                for token in ancestors:
                    if token.pos_ in ['PRON', 'PROPN', 'NOUN']:
                        syntactic_head_token = token
                        break
                    elif token.pos_ in ['VERB', 'AUX']:
                        syntactic_head_token = token
                        break

                if syntactic_head_token is None:
                    for token in ancestors:
                        if token.pos_ in ['NUM', 'ADJ']:
                            syntactic_head_token = token
                            break

        # postprocessing for CC and CONJ -- exclude child arcs with CC or CONJ
        span_start, span_end = self.get_final_span(syntactic_head_token, head_token)

        return span_start, span_end


    @staticmethod
    def create_tables():
        db = DDL('LU')
        db.create_tables()
        db.close()

    def close(self):
        self.ma_con.commit()
        self.ma_con.close()


if __name__ == '__main__':
    test = CSDS2Master()
    test.generate_database()


