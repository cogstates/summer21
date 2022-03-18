#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import getpass
import math
import operator
import os
import itertools
import random
from itertools import islice
from functools import reduce

from datasets import Dataset, DatasetDict, ClassLabel
from more_itertools import seekable
import nlpaug.augmenter.word as naw
import pandas as pd

from jadoch.data import costep
from jadoch.data.costep import language, contains, starts_with, some
from jadoch.core.app import harness, slurmify
from jadoch.core.functional import ilen, save_iter


german = language("german")
english = language("english")


# Hopefully this ensures it's the correct "ja".
ignores = map(
    contains,
    [
        "sagt ja",
        "sagen ja",
        "sage ja",
        "sagten ja",
        "sagte ja",
        "ja oder",
        "ja zum",
        "ja zur",
        "ja zu",
    ],
)
fltr = german(contains("ja") & ~starts_with("ja") & ~some(*ignores)) & english(
    ~contains("yes")
)


def labelify(label):
    def func(val):
        return (val, label)

    return func


def augmentify(model_path="distilbert-base-uncased", action="substitute"):
    model = naw.ContextualWordEmbsAug(model_path=model_path, action=action)

    def func(txt):
        return model.augment(txt)

    return func


def mapify(functions, iterable):
    fns = iter(functions)
    for fn in fns:
        iterable = map(fn, iterable)
    return iterable


def search(fltr, label, fn=lambda x: x):
    return mapify(
        (operator.itemgetter("english"), fn, labelify(label)),
        filter(fltr, costep.sentences("english", "german")),
    )


def split(itr, pct):
    items = list(itr)
    idx = round(len(items) * pct)
    return items[:idx], items[idx:]


def partition(iterable, sizes):
    it = iter(iterable)

    for size in sizes:
        if size is None:
            yield list(it)
            return
        else:
            yield list(islice(it, size))


# def generate(fltr, limit=None):
#     jas = itertools.chain(
#         *[search(fltr, "ja")] + [search(fltr, "ja", augmentify()) for _ in range(145)]
#     )
#     nas = search(~fltr, "na")
#     train_jas, test_jas = split(islice(jas, limit), 0.8)
#     train_nas, test_nas = split(islice(nas, limit), 0.8)
#     training_data = train_jas + train_nas
#     testing_data = test_jas + test_nas
#     random.shuffle(training_data)
#     random.shuffle(testing_data)
#     class_label = ClassLabel(num_classes=2, names=["na", "ja"])  # XXX: ???
#     reshape = lambda dt: {
#         "text": [tup[0] for tup in dt],
#         "label": list(map(class_label.str2int, [tup[1] for tup in dt])),
#     }
#     return DatasetDict(
#         {
#             "train": Dataset.from_dict(reshape(training_data)),
#             "test": Dataset.from_dict(reshape(testing_data)),
#         }
#     )

# NOTE: This method will "undersample" data.
# def generate(fltr, pct_train, pct_test, ja_limit=None, na_limit=None):
#     ja_limit = ja_limit or ilen(search(fltr, "ja"))
#     na_limit = na_limit or ilen(search(~fltr, "na"))
#     jas = search(fltr, "ja")
#     nas = search(~fltr, "na")
#     train_jas, test_jas = split(islice(jas, ja_limit), 0.8)
#     train_nas, test_nas = partition(
#         nas,
#         [
#             int(len(train_jas) * (1 - pct_train) / pct_train)
#             if pct_train
#             else int(na_limit * 0.8),
#             int(len(test_jas) * (1 - pct_test) / pct_test)
#             if pct_test
#             else int(na_limit * 0.2),
#         ],
#     )
#     training_data = train_jas + train_nas
#     testing_data = test_jas + test_nas
#     random.shuffle(training_data)
#     random.shuffle(testing_data)
#     class_label = ClassLabel(num_classes=2, names=["na", "ja"])  # XXX: ???
#     reshape = lambda dt: {
#         "text": [tup[0] for tup in dt],
#         "label": list(map(class_label.str2int, [tup[1] for tup in dt])),
#     }
#     return DatasetDict(
#         {
#             "train": Dataset.from_dict(reshape(training_data)),
#             "test": Dataset.from_dict(reshape(testing_data)),
#         }
#     )


def repeat(iterable, n=None):
    """Repeat an iterable until n items have been yielded.

    >>> list(repeat('AB', 3))
    [(0, 'A'), (0, 'B'), (1, 'A'), (1, 'B'), (2, 'A'), (2, 'B')]

    """
    it = seekable(iter(iterable))
    cnt = 0
    while cnt < n:
        for item in it:
            yield item
            cnt += 1
            if cnt >= n:
                break
        it.seek(0)


# NOTE: This method will "oversample" (or "repeat") data.
def generate(fltr, pct_train, pct_test, ja_limit=None, na_limit=None):
    ja_limit = ja_limit or ilen(search(fltr, "ja"))
    na_limit = na_limit or ilen(search(~fltr, "na"))
    jas = search(fltr, "ja")
    nas = search(~fltr, "na")
    train_jas, test_jas = split(islice(jas, ja_limit), 0.8)
    train_jas = list(
        repeat(
            train_jas,
            math.floor((na_limit * 0.8 * pct_train) / (1 - pct_train))
            if pct_train
            else int(ja_limit * 0.8),
        )
    )
    test_jas = list(
        repeat(
            test_jas,
            (na_limit * 0.2 * pct_test) / (1 - pct_test)
            if pct_test
            else int(ja_limit * 0.2),
        )
    )
    train_nas, test_nas = split(islice(nas, na_limit), 0.8)

    training_data = train_jas + train_nas
    testing_data = test_jas + test_nas
    random.shuffle(training_data)
    random.shuffle(testing_data)
    class_label = ClassLabel(num_classes=2, names=["na", "ja"])  # XXX: ???
    reshape = lambda dt: {
        "text": [tup[0] for tup in dt],
        "label": list(map(class_label.str2int, [tup[1] for tup in dt])),
    }
    return DatasetDict(
        {
            "train": Dataset.from_dict(reshape(training_data)),
            "test": Dataset.from_dict(reshape(testing_data)),
        }
    )


def main(ctx):
    ctx.parser.add_argument("--pct-train", type=float)
    ctx.parser.add_argument("--pct-test", type=float)
    ctx.parser.add_argument(
        "-o", "--outdir", default=f"/gpfs/scratch/{getpass.getuser()}/data/"
    )
    ctx.parser.add_argument("-i", "--info", action="store_true")
    args = ctx.parser.parse_args()

    pct_train = int(args.pct_train * 100) if args.pct_train else "natural"
    pct_test = int(args.pct_test * 100) if args.pct_test else "natural"
    path = os.path.join(
        args.outdir, f"english-{pct_train}-train-{pct_test}-test-repeat"
    )
    if args.info:
        ctx.log.info("reading from %s", path)
        dt = DatasetDict.load_from_disk(path)
    else:
        ctx.log.info("writing to %s", path)
        dt = generate(fltr, args.pct_train, args.pct_test)
        dt.save_to_disk(path)
    df = pd.DataFrame(
        {
            lbl: {
                "jas": sum(dt[lbl]["label"]),
                "nas": dt[lbl].num_rows - sum(dt[lbl]["label"]),
                "total": dt[lbl].num_rows,
            }
            for lbl in ("train", "test")
        }
    ).T
    ctx.log.info("\n" + str(df))
    # generate(fltr).save_to_disk("/gpfs/scratch/asoubki/data/english-balanced-aug")


if __name__ == "__main__":
    harness(main)
