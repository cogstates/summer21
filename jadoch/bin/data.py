#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import operator
import itertools
import random
from itertools import islice
from functools import reduce

from datasets import Dataset, DatasetDict, ClassLabel
import nlpaug.augmenter.word as naw

from jadoch.data import costep
from jadoch.data.costep import language, contains, starts_with, some
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


def generate(fltr, limit=None):
    jas = itertools.chain(
        *[search(fltr, "ja")] + [search(fltr, "ja", augmentify()) for _ in range(145)]
    )
    nas = search(~fltr, "na")
    train_jas, test_jas = split(islice(jas, limit), 0.8)
    train_nas, test_nas = split(islice(nas, limit), 0.8)
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


def main():
    generate(fltr).save_to_disk("/gpfs/scratch/asoubki/data/english-balanced-aug")


if __name__ == "__main__":
    main()
