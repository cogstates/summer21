#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summarizes label counts given an output directory from `extract.py`.

Example Usage:
    $ ls data/proc/CMN/
    > dev.jsonl  test.jsonl  train.jsonl
    $ ./bin/count.py data/proc/CMN/
    > CB    NA   ROB  NCB  total
    > train   7235  5077   961  185  13458
    > dev     1049   524   152   12   1737
    > test    2271  1355   193   70   3889
    > total  10555  6956  1306  267  19084
"""
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections import Counter
from operator import itemgetter

import pandas as pd
from more_itertools import flatten


def main():
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("indir", help="directory path to summarize")
    args = parser.parse_args()

    data = {}
    for split in ("train", "dev", "test"):
        df = pd.read_json(f"./data/proc/CMN/{split}.jsonl", lines=True)
        data[split] = Counter(map(itemgetter("label"), flatten(df.targets.tolist())))
    df = pd.DataFrame(data).T
    df = df.rename(columns={0: "NA", 1: "ROB", 2: "NCB", 3: "CB"})
    df["total"] = df.sum(axis=1)
    df = df.T
    df["total"] = df.sum(axis=1)
    df = df.T
    print(df)


if __name__ == "__main__":
    main()
