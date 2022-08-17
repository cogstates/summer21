#!/usr/bin/env python
# -*- coding: utf-8 -*-
import functools
import glob
import itertools
import json
import re
import os
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from operator import attrgetter
from typing import List
from xml.etree import ElementTree

import spacy
import numpy as np


labels_ldc = {
    "CB": 3, 
    "NCB": 2, 
    "ROB": 1, 
    "NA": 0
}
language_models = {
    "en": "en_core_web_lg",
    "es": "es_core_news_lg",
    "zh": "zh_core_web_lg"
}
rgx = r"\((ann-\d+)\)"


@dataclass
class Annotation:
    aid: str
    source_uid: str
    offset: int
    length: int
    belief_type: str
    text: str

    @property
    def marker(self):
        return "(" + self.aid + ")"

    @classmethod
    def from_xml(cls, xml):
        return cls(
            aid=xml.attrib["id"],
            source_uid=xml.attrib["source_uid"],
            offset=int(xml.attrib["offset"]),
            length=int(xml.attrib["length"]),
            belief_type=xml.find("belief_type").text,
            text=xml.find("annotation_text").text
        )


@dataclass
class Target:
    span1: List[int]
    span_text: str
    label: float

    @classmethod
    def from_tokens(cls, text, tkns, lang, ann):
        from itertools import accumulate
        from more_itertools import first, locate

        try:
            atext = "".join(text.split())
            atkns = [tkn.text for tkn in get_spacy(lang)(ann.text.strip())]
            adx = len(re.sub(rgx, "", re.split(fr"\({ann.aid}\)", atext.replace(" ", ""))[0]))
            ldx = first(locate(accumulate(map(len, tkns)), lambda v: v > adx))
            rdx = ldx + len(atkns)
            assert "".join(tkns)[adx:adx + len(ann.text.strip())] == ann.text.strip()
            assert "".join(atkns) in "".join(tkns[ldx:rdx])
            return cls(
                    span1=[ldx, rdx],
                    span_text=" ".join(tkns[ldx:rdx]),
                    label=labels_ldc[ann.belief_type]
            )
        except:
            import ipdb; ipdb.set_trace()
        

cntr = itertools.count(0)


@dataclass
class Sentence:
    idx: int
    file_idx: int
    text: str
    targets: List[Target]

    @classmethod
    def from_text_and_annotations(cls, text, anns, lang):
        from functools import partial

        tkns = [tkn.text.replace(" ", "") for tkn in get_spacy(lang)(re.sub(rgx, "", text))]
        tkns = [tkn.strip() for tkn in tkns if tkn.strip()]
        return cls(
            idx=next(cntr),
            file_idx=None,
            text=" ".join(tkns),
            targets=list(map(partial(Target.from_tokens, text, tkns, lang), anns))
        )


@functools.lru_cache()
def get_spacy(lang):
    model = language_models[lang]
    try:
        return spacy.load(model)
    except:
        spacy.cli.download(model)
        return spacy.load(model)


def add_markers(raw, anns):
    moffset = 0
    for ann in sorted(anns, key=attrgetter("offset")):
        ldx, rdx = moffset + ann.offset, moffset + ann.offset + ann.length
        assert raw[ldx:rdx] == ann.text
        moffset += len(ann.marker)
        raw = raw[:ldx] + ann.marker + raw[ldx:]
    return raw


# XXX: This is an extremely janky thing. Instead of keeping track of offsets
#      while stripping the xml, I just tag where the annotation is which
#      sometimes confuses the sentence segmenter. To deal with this I check
#      for any sentences ending with an annotation tag and merge them with the
#      sentence in front of them. It seems to mostly work but it is bad.
def merge_sentences(snts):
    ret = []
    for snt in snts:
        if ret and re.findall(rgx + "$", ret[-1]):
            ret[-1] += snt
        else:
            ret.append(snt)
    return ret


def process_one(src_path, ann_path, lang):
    ret = []

    anns = list(map(Annotation.from_xml, ElementTree.parse(ann_path).findall(".//annotation")))
    with open(src_path, "r", encoding="utf-8") as fd:
        raw = fd.read()
    root = ElementTree.fromstring("<root>" + add_markers(raw, anns) + "</root>")
    text = "".join([txt.strip() for txt in root.itertext() if txt.strip()])
    sentences = [snt.text.strip() for snt in get_spacy(lang)(text).sents if snt.text.strip()]
    sentences = merge_sentences(sentences)

    # Get all annotations that apply to each sentence.
    annmap = {ann.aid: ann for ann in anns}
    for snt in sentences:
        aids = re.findall(rgx, snt)
        if not aids:
            continue
        ret.append(
            Sentence.from_text_and_annotations(
                snt,
                list(map(lambda k: annmap[k], aids)),
                lang
            )
        )
    # NOTE: Skips sentences 100 and longer because they don't play nice with jiant.
    return filter(lambda s: len(s.text.split()) < 100, ret)


def process(inpaths, outpath, lang):
    with open(outpath, "w") as fd:
        for src, ann in inpaths:
            print(src, ann)
            for snt in process_one(src, ann, lang):
                fd.write(json.dumps(asdict(snt)) + "\n")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--indir", required=True)
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("-l", "--language", choices=language_models.keys(), required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    srcs = sorted(glob.glob(os.path.join(args.indir, "source", "*")))
    anns = sorted(glob.glob(os.path.join(args.indir, "annotation", "*")))
    files = list(zip(srcs, anns))
    train, dev, test = np.split(files, [int(len(files) * 0.7), int(len(files) * 0.8)])
    print("Splits:", list(map(len, [train, dev, test])))
    process(train, os.path.join(args.outdir, "train.jsonl"), args.language)
    process(dev, os.path.join(args.outdir, "dev.jsonl"), args.language)
    process(test, os.path.join(args.outdir, "test.jsonl"), args.language)


if __name__ == "__main__":
    main()
