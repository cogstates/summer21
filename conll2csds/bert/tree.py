UNKNOWN = "UNK"
ROOT = "ROOT"
NULL = "NULL"
NONEXIST = -1
# inbuilt lib imports:
from typing import List, Dict, Tuple, Any, NamedTuple
import math
from CSDS.csds import CSDS, CSDSCollection

import numpy as np
from tqdm import tqdm
import numpy as np
from collections import defaultdict

import copy

class ConllCSDS(CSDS):
    head_idx = -1

    def __init__(
            self, this_text, head_idx, this_belief, this_head="",
            this_doc_id=-1, this_sentence_id=-1
    ):
        self.doc_id = this_doc_id
        self.sentence_id = this_sentence_id
        self.text = this_text
        self.head_idx = head_idx
        self.belief = this_belief
        self.head = this_head

    def get_marked_text(self):
        return self.text



class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self,child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children>0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

def head_to_tree(head):
    """
    Convert a sequence of head indexes into a tree object.
    """
    head = sorted(head)


    root = None
    nodes = [Tree() for _ in head]


    for i in range(len(nodes)):
        h = head[i]
        nodes[i].idx = i
        nodes[i].dist = -1
        if h == 0:
            root = nodes[i]
        else:
            nodes[h-1].add_child(nodes[i])


    return root


def tree_to_adj(sent_len, tree, not_directed=True):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)
    # ret = np.ones((sent_len, sent_len), dtype=np.float32)


    length = ret.shape[0]

    queue = [tree]


    idx = []

    while len(queue) > 0:
        t, queue = queue[0], queue[1:]


        idx += [t.idx]

        for c in t.children:

            ret[t.idx, c.idx] = 1

        queue += t.children

    if not_directed:
        ret = ret + ret.T

    ret = ret + np.eye(sent_len)

    return ret
train_path = '/conll2csds/factbank_v1/train.conll'
dev_path = '/conll2csds/factbank_v1/dev.conll'
test_path = '/conll2csds/factbank_v1/test.conll'

def get_spans(heads):
    spans = []
    for i in heads:
        vals, inverse, count =np.unique(i, return_inverse=True,
                                      return_counts=True)
        idx_vals_repeated = np.where(count > 1)[0]
        vals_repeated = vals[idx_vals_repeated]

        rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
        _, inverse_rows = np.unique(rows, return_index=True)
        res = np.split(cols, inverse_rows[1:])
        spans.append(res)
    return spans


def read_conll_data(data_file_path: str):
    """
    Reads Sentences and Trees from a CONLL formatted data file.

    Parameters
    ----------
    data_file_path : ``str``
        Path to data to be read.
    """
    csds = CSDSCollection("")

    sentences = []
    heads = []
    sp = []
    labels = []
    with open(data_file_path, 'r') as file:
        sentence_tokens = []
        h = []
        s = []
        for line in tqdm(file):
            line = line.strip()
            array = line.split('\t')
            if len(array) < 7:
                if sentence_tokens:
                    sentences.append(sentence_tokens)
                    sentence_tokens = []
                heads.append(h)
                labels.append(s)

                s = []
                h = []
            else:
                word = array[1]
                belief = (array[2])
                s.append(belief)
                head = int(array[4])
                pos = array[5]
                h.append(head)
                token = word
                sentence_tokens.append(token)
    corpus = []
    spans = get_spans(heads)
    bv = []
    for sent_index, i in enumerate(labels):
        for index, ii in enumerate(i):
            if ii != '_':
                sform = sentences[sent_index]
                print(sentences[sent_index])
                head_temp = sentences[sent_index][int(heads[sent_index][index])]
                replacement = "* " + sentences[sent_index][int(heads[sent_index][index])] + " *"
                (sform[int(heads[sent_index][index])]) = replacement
                joined = ' '.join(sform)
                print(sform)
                corpus.append(
                    (joined, int(heads[sent_index][index]), float(ii), sentences[sent_index][int(heads[sent_index][index])]))
                head_temp = (sform[int(heads[sent_index][index])])


    for sentence_id, sample in enumerate(corpus):
        csds.add_labeled_instance(ConllCSDS(*sample, 0, sentence_id))

    text = []
    belief = []
    for instance in csds.get_next_instance():
        text.append(instance.get_marked_text())
        belief.append(instance.get_belief())

    final = []
    final_bv = []
    for i in sp:
        for ii in i:
            final.append(' '.join(ii))

    for i in bv:
        for ii in i:
            final_bv.append(float(ii))

    return sentences, heads, final, final_bv



sentence, heads, spans, bv = read_conll_data(train_path)


#print(get_spans(heads)[0])
#for i in get_spans(heads)[0]:

