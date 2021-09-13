import sys

import numpy as np

sys.path.append('../')
import os
from spacy.lang.en import English
from collections import defaultdict
#from bs4 import BeautifulSoup
import logging
#from HTMLParser import HTMLParser
from unidecode import unidecode
from pprint import pprint
logging.basicConfig(level = logging.DEBUG)
import spacy
from spacy.matcher import Matcher
from spacy.tokenizer import Tokenizer
import re
special_cases = {":)": [{"ORTH": ":)"}]}
prefix_re = re.compile(r'''^[\[\(""']''')
suffix_re = re.compile(r'''[\]\)/""']$''')
infix_re = re.compile(r'''[-~]''')
simple_url_re = re.compile(r'''^https?://''')

def prevent_sbd(doc):
    """ Ensure that SBD does not run on tokens inside quotation marks and brackets. """
    quote_open = False
    bracket_open = False
    can_sbd = True
    for token in doc:
        # Don't do sbd on these tokens
        if not can_sbd:
            token.is_sent_start = False

        # Not using .is_quote so that we don't mix-and-match different kinds of quotes (e.g. ' and ")
        # Especially useful since quotes don't seem to work well with .is_left_punct or .is_right_punct
        if token.text == '"':
            quote_open = False if quote_open else True
        elif token.is_bracket and token.is_left_punct:
            bracket_open = True
        elif token.is_bracket and token.is_right_punct:
            bracket_open = False

        can_sbd = not (quote_open or bracket_open)

    return doc


nlp = spacy.load("en_core_web_md")



class Factbank:
    """
    Reads factbank annotation files, producing (as class members):
    - CONLL string represeting parse and factuality annotations.
    - FB values from "FactBank: a corpus annotated with event factuality" (Sauri, 2009)
    Committed Values:
    CT+ - According to the source, it is certainly the case that X.
    Our value: 3.0
    PR+ - According to the source, it is probably the case that X.
    Our value: 2.0
    PS+ - According to the source, it is possibly the case that X.
    Our value: 1.0
    CT- - According to the source, it is certainly not the case that X.
    Our value: -3.0
    PR- - According to the source it is probably not the case that X.
    Our value: -2.0
    PS- - According to the source it is possibly not the case that X.
    Our value: -1.0
    (Partially) Uncommitted Values:
    CTu - The source knows whether it is the case that X or that not X.
    Our value: 0.0
    Uu  - The source does not know what is the factual status of the
    event, or does not commit to it.
    Our value: 0.0
    """
    def __init__(self, fb_factvalue, tokens_tml, sentence_threshold = 3):
        """
        fb_factvalue - the patht to fb_factvalue.txt in Factbank
        tokens_tml - the patht to tokens_tml in Factbank
        sentence_threshold - Filter sentences shorter than this number
                             (This helps in removal of boiler plate sentences)
        """
        self.fact_annots = self.load_fact_annots(fb_factvalue)
        self.sentence_threshold = sentence_threshold
        self.conversion_dic = {
            "CT+": 3.0,
            "PR+": 2.0,
            "PS+": 1.0,
            'CTu': 0.0,
            'Uu' : 0.0,
            "PS-": -1.0,
            "PR-": -2.0,
            "CT-": -3.0
        }
        self.conll_txt = self.convert(tokens_tml)

    def __str__(self):
        return self.conll_txt

    def load_fact_annots(self, fb_factvalue):
        """
        Given an fb_factvalue file, returns a dictionary:
        filename -> sent_number -> entity -> source -> factuality value
        """
        ret = {}
        for line in open(fb_factvalue):
            line = line.strip()
            if not line:
                continue
            filename, sent_id, fvid, eId, \
            eiId, relSourceId, eText, \
            relSourceText, factValue = [eval(v) for v in line.split('|||')]
            if factValue:
                keys = [filename, sent_id, eId, relSourceText]
                ddict_app(ret, factValue, keys)
                assert len(ddict_get(ret, keys)) == 1,\
                    "More than one value by source: {}".format((keys, ddict_get(ret, keys)))
        return ret

    def to_float(self, fact_val):
        """
        Return a float factual value for a given textual input
        """
        return self.conversion_dic[fact_val]

    def consolidate_fact_value(self, filename, sent_number, entity_id):
        """
        Return the *float* factuality value of a given set of keys.
        Consolidates over the possibly multiple sources
        """
        opts = ddict_get(self.fact_annots,
                         [filename, sent_number, entity_id])




        commit_opts = sorted([list(opt)[0] for opt in opts.values()
                              if ('+' in list(opt)[0]) or ('-' in list(opt)[0])],
                             reverse = True)
        c = ((str(list(opts)[0])))
        c = c.replace("{", "")
        c = c.replace("}", "")
        c = c.replace("'", "")
        t = list(opts.values())[0]
        t = (''.join(t))



   # if there are no commited values, return the first option
        return self.to_float(commit_opts[0]) if commit_opts \
            else self.to_float(t)



    def convert(self, token_tml):
        """
        Convert a token_tml file to conll format
        """
        sents = []
        cur_sent = []
        last_sent = -1
        for line in open(token_tml):
            line = line.strip()
            if not line:
                continue
            fn, sent_id, tok_id, \
            surface_form, tmlTag, tmlTagId, tmlTagLoc = [eval(v) for v in line.split('|||')]
            cur_ent = [tok_id,
                       surface_form,
                       self.consolidate_fact_value(fn, sent_id, tmlTagId) \
                       if (tmlTag == 'EVENT')\
                          else  "_"]

            if sent_id != last_sent:
                if cur_sent:
                    toks = nlp(str(" ".join([word[1] for word in cur_sent])))
                    #toks = toks.replace('"','')
                    #print(toks)
                    dep_feats = self.get_dep_feats(toks, cur_sent)
                    sents.append([fb_feat + dep_feat
                                  for (fb_feat, dep_feat) in zip(cur_sent, dep_feats)])
                cur_sent = [cur_ent]
            else:
                cur_sent.append(cur_ent)
            last_sent = sent_id

        return '\n\n'.join(['\n'.join(['\t'.join(map(str, word))
                                       for word in sent])
                            for sent in sents
                            if len(sent) > self.sentence_threshold]) + "\n\n"  # filter short sentences

    def get_dep_feats(self, toks, sent):
        """
        Return the required features to complete the conll format:
        1. POS
        2. Head index
        3. Head relations
        4. Lemma
        """
        self.align(toks, sent)
        assert(len(toks) == len(sent))  # After alignment there should be the of equal lengths
        return [[tok.tag_,
                 str(tok.head.i),
                 tok.dep_,
                 unidecode(tok.lemma_)] for tok in toks]

    def align(self, toks, sent):
        """
        Match between the spacy tokens in toks to the words in sent
        Might merge tokens in spacy in-place.
        """
        toks_ind = 0
        sent_ind = 0
        ret = []
        while sent_ind < len(sent):
            #logging.debug("sent_ind = {}, toks_ind = {}".format(sent_ind, toks_ind))
            cur_tok = str(toks[toks_ind])
            cur_word = sent[sent_ind][1]
            #logging.debug("{} vs. {}".format(cur_tok, cur_word))
            #logging.debug("flag = {}".format(cur_word.endswith(cur_tok)))
            if cur_tok == "." and cur_word == ". . .":
                with toks.retokenize() as retokenizer:
                    retokenizer.merge(toks[toks_ind: toks_ind + 3])
                continue

            if cur_tok.isspace() and cur_word.isspace():
                toks_ind += 1
                sent_ind += 1
                continue

            if (cur_tok == cur_word) or \
                    (cur_word.endswith(cur_tok) and \
                     (toks_ind >= (len(toks) - 1) or ((cur_tok + str(toks[toks_ind + 1])) not in cur_word))):
                toks_ind += 1
                sent_ind += 1

            elif cur_tok in cur_word:
                print("merging: {}".format(toks[toks_ind : toks_ind + 2]))
                with toks.retokenize() as retokenizer:
                    retokenizer.merge(toks[toks_ind: toks_ind + 2])
                    print(toks)
            else:
                print(toks)
                raise Exception("Unknown case: {}".format((toks,
                                                           cur_tok,
                                                           cur_word)))
        assert (toks_ind == len(toks))
        return ret


def ddict_app(d, val, args):
    """
    Append (to a set) the value 'val' to an arbitrary nesting in dictionary 'd'
    """
    cur = d
    for k in args[:-1]:
        cur[k] = cur.get(k, {})
        cur = cur[k]
    last_k = args[-1]
    cur[last_k] = cur.get(last_k, set())
    cur[last_k] = cur[last_k].union([val])

def ddict_get(d, keys):
    """
    Return a value from an arbitrary nesting level in dictionary 'd'
    """
    cur = d
    for k in keys:
        cur = cur[k]
    return cur

print(Factbank("/home/john/PycharmProjects/summer21/data/annotation/fb_factValue.txt", "/home/john/PycharmProjects/summer21/data/annotation/tokens_tml.txt"))