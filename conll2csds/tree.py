UNKNOWN = "UNK"
ROOT = "ROOT"
NULL = "NULL"
NONEXIST = -1
# inbuilt lib imports:
from typing import List, Dict, Tuple, Any, NamedTuple
import math

import numpy as np
from tqdm import tqdm
import numpy as np
from collections import defaultdict

import copy


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
train_path = '/Users/john/PycharmProjects/summer21/conll2csds/factbank_v1/train.conll'
dev_path = '/Users/john/PycharmProjects/summer21/conll2csds/factbank_v1/dev.conll'
test_path = '/Users/john/PycharmProjects/summer21/conll2csds/factbank_v1/test.conll'

def read_conll_data(data_file_path: str):
    """
    Reads Sentences and Trees from a CONLL formatted data file.

    Parameters
    ----------
    data_file_path : ``str``
        Path to data to be read.
    """
    sentences = []
    trees = []
    heads = []
    spans = []
    with open(data_file_path, 'r') as file:
        sentence_tokens = []
        h = []
        for line in tqdm(file):
            line = line.strip()
            array = line.split('\t')
            if len(array) < 7:
                if sentence_tokens:
                    sentences.append(sentence_tokens)
                    sentence_tokens = []
                heads.append(h)
                '''
                for h in heads:
                    h = np.array(h)

                    vals, idx_start, count = np.unique(h, return_counts=True, return_index=True)
                    spans.append(vals)
                '''
                h = []
            else:
                word = array[1]
                belief = (array[2])
                head = int(array[4])
                pos = array[5]
                if pos == 'ROOT':
                    head = 0
                    h.append(head)
                else:
                    h.append(head)
                token = [(word, belief,
                              head)]
                sentence_tokens.append(token)




    return sentences, heads, spans


sentence, heads, spans = read_conll_data(train_path)
print(
    heads[1]
)
print(tree_to_adj(len(heads[1]), head_to_tree(heads[1])))