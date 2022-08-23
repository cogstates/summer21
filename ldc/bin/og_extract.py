#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import glob
import json
import os
import re
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize


labels_ldc = {
    'CB': 3,
    'NCB': 2,
    'ROB': 1,
    'NA': 0
}


def get_data(file_annotation, file_source):
    with open(file_source, 'r', encoding='utf-8') as f:
        text = f.read().replace("*", "#")

    mytree = ET.parse(file_annotation)
    root = mytree.getroot()

    offset_map = {}
    offsets = []
    labels = []
    heads = []
    lengths = []

    for annotation in root.findall("annotation"):
        belief = annotation.find('belief_type').text
        offset = int(annotation.get('offset'))
        length = int(annotation.get('length'))
        annotation_text = annotation.find('annotation_text').text.replace("*", "#")
        # offsets_to_lengths[int(annotation.get('offset'))] = int(annotation.get('length'))
        offset_map[offset] = (length, belief, annotation_text)
        offsets.append(offset)
        labels.append(length)
        heads.append(annotation_text)
    
    offsets = sorted(list(set(offsets)))

    # Rather than applying the stars in place, let's do it sentence by sentence
    for i in range(len(offsets)):
        offset = offsets[i] + i
        text = text[:offset] + '*' + text[offset:]

    # After heads labeled, we want to tokenize
    # print(text)

    parser = BeautifulSoup(text, 'html.parser')
    text = parser.get_text()

    # text = text.replace('\n', '')
    # Finally, we use nltk sentencizer to obtain the sentences.
    sentences = sent_tokenize(text, language='english')

    sentences = [sentence.replace('\n', '') for sentence in sentences if '*' in sentence]

    # counter = sum([sentence.count('*') for sentence in sentences])
    # print(sentences)
    # print(len(sentences))
    # print(counter)
    # print(len(labels))

    offset_map = collections.OrderedDict(sorted(offset_map.items()))
    lengths = [offset_map[offset][0] for offset in offset_map]
    labels = [offset_map[offset][1] for offset in offset_map]
    heads = [offset_map[offset][2] for offset in offset_map]

    # print(lengths)
    length_counter = 0
    text_dict = {}
    texts = []
    # lbls = []
    CLEANR = re.compile('<.?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    for sentence in sentences:
        original_sentence = sentence.replace('*', '')
        targets = []
        while '*' in sentence:
            # print(sentence)
            star_index = sentence.find('*')
            if star_index >= 0:
                # print(lengths[length_counter], heads[length_counter], labels[length_counter], len(sentence), star_index)

                if length_counter < len(lengths) and (star_index + 1 + lengths[length_counter] < len(sentence)):
                    if sentence[star_index + 1:star_index + 1 + lengths[length_counter]] == heads[length_counter]:
                        # print(sentence)
                        target_dict = {}
                        words = sentence.split()
                        
                        words = [re.sub(r'[.,?!]', '', i) for i in words]
                        # print(words)
                        if '*' + heads[length_counter] in words:
                            span_start = words.index('*' + heads[length_counter])
                        elif '*' + heads[length_counter] + '.' in words:
                            span_start = words.index('*' + heads[length_counter] + '.')
                        elif '*' + heads[length_counter] + ',' in words:
                            span_start = words.index('*' + heads[length_counter] + ',')
                        else:
                            sentence = sentence.replace('*', '', 1)
                            length_counter += 1
                            continue

                        target_dict['span1'] = [span_start, span_start + 1]
                        target_dict['label'] = labels_ldc[labels[length_counter]]
                        target_dict['span_text'] = heads[length_counter]

                        targets.append(target_dict)
                        sentence = sentence.replace('*', '', 1)
                        length_counter += 1
                    else:
                        length_counter += 1
                else:
                    # length_counter += 1
                    # print("test")
                    break

        text_dict[original_sentence] = targets

    for sentence in text_dict:
        if len(text_dict[sentence]) > 0:
            texts.append({'text': sentence, 'targets': text_dict[sentence]})
    
    return texts


def process(inpaths, outpath):
    with open(outpath, "w") as fd:
        for src, ann in inpaths:
            print(src, ann)
            for snt in get_data(ann, src):
                fd.write(json.dumps(snt) + "\n")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--indir", required=True)
    parser.add_argument("-o", "--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    test = list(zip(
        sorted(glob.glob(os.path.join(args.indir, "source", "test", "*"))),
        sorted(glob.glob(os.path.join(args.indir, "annotation", "test", "*")))
    ))
    srcs = sorted(glob.glob(os.path.join(args.indir, "source", "train", "*")))
    anns = sorted(glob.glob(os.path.join(args.indir, "annotation", "train", "*")))
    files = list(zip(srcs, anns))
    train, dev = np.split(files, [int(len(files) * (8 / 9))])
    print("Splits:", list(map(len, [train, dev, test])))
    process(train, os.path.join(args.outdir, "train.jsonl"))
    process(dev, os.path.join(args.outdir, "dev.jsonl"))
    process(test, os.path.join(args.outdir, "test.jsonl"))


if __name__ == "__main__":
    main()
