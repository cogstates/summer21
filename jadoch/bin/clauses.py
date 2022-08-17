#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""An example script"""
import getpass
import json
import os
import random
from collections import Counter
from functools import lru_cache
from itertools import chain, islice, starmap
from operator import attrgetter, itemgetter

import pycountry
import spacy
import torch
import transformers
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from simalign import SentenceAligner

from jadoch.core.context import Context
from jadoch.core.app import harness
from jadoch.core.functional import save_iter
from jadoch.data import costep
from jadoch.data.costep import contains, language, some, starts_with


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


lang_to_model = {
#      "en": "en_core_web_trf",
#      "de": "de_dep_news_trf"
#  } if torch.cuda.is_available() else {
    "en": "en_core_web_lg",
    "de": "de_core_news_lg"
}


@lru_cache
def _get_spacy(lang):
    try:
        return spacy.load(lang_to_model[lang])
    except:
        spacy.cli.download(lang_to_model[lang])
        return spacy.load(lang_to_model[lang])


def get_spacy(lang):
    return _get_spacy(pycountry.languages.lookup(lang).alpha_2)


@lru_cache
def get_aligner(model="bert"):
    return SentenceAligner(model=model, matching_methods="i")


def mapify(functions, iterable):
    fns = iter(functions)
    for fn in fns:
        iterable = map(fn, iterable)
    return iterable


def spacify(dct):
    for lang in dct:
        if lang == "meta":
            continue  # Skip
        dct[lang] = get_spacy(lang)(dct[lang])
    return dct


def tokenify(dct):
    for lang in dct:
        if lang == "meta":
            continue  # Skip
        dct[lang] = " ".join(map(str, get_spacy(lang)(dct[lang])))
    return dct


def get_children(tkn):
    return sorted(
        [tkn] + list(chain.from_iterable(map(get_children, tkn.children))),
        key=attrgetter("i")
    )


def clausify(dct):
    from more_itertools import first, flatten, one
    from collections import defaultdict

    dct["clauses"] = {}
    # Extract the "ja" head from the german sentence.
    ja = first(filter(lambda s: s.text.lower() == "ja", dct["german"]))  # XXX: How to know which one?
    head = ja.head
    while True:
        if head.pos_ in ("VERB", "AUX"):
            break
        if head == head.head:
            break  # Reached the top.
        head = head.head
    if head.head.pos_ == "AUX" and head.dep_ == "oc":
        head = head.head
    dct["clauses"]["german"] = " ".join(map(attrgetter("text"), get_children(head)))
    # Generate an alignment.
    alignment = get_aligner().get_word_aligns(
        list(map(attrgetter("text"), dct["german"])),
        list(map(attrgetter("text"), dct["english"])),
    )["itermax"]
    alignmap = defaultdict(list)
    for ddx, edx in alignment:
        alignmap[dct["german"][ddx]].append(dct["english"][edx])
    dct["alignment"] = dict(alignmap)
    # Resolve the alignment with the english text.
    matches = []
    for match in alignmap[head]:
        # TODO: Check that match.pos_ is in ("VERB", "AUX").
        if match.pos_ == "AUX" and not list(match.children):
            match = match.head  # If it is a leaf and tagged as AUX go up one.
        matches.append(match)
    matches = sorted(list(set(flatten(map(get_children, matches)))), key=attrgetter("i"))
    dct["clauses"]["english"] = " ".join(map(attrgetter("text"), matches))
    # Return the final result.
    return dct


# XXX: Set the random seed.
# This is used for the "na" examples.
def random_clausify(dct):
    dct["clauses"] = {}
    dct["head"] = {}
    head = random.choice(dct["english"])
    dct["head"]["start"] = head
    while True:
        if head.pos_ in ("VERB", "AUX"):
            break
        if head == head.head:
            break  # Reached the top.
        head = head.head
    if head.head.pos_ == "AUX" and head.dep_ == "oc":
        head = head.head
    dct["head"]["end"] = head
    dct["clauses"]["english"] = " ".join(map(attrgetter("text"), get_children(head)))
    return dct


def labelify(label):
    def func(dct):
        return dct["clauses"]["english"], label
    return func


def cache(path):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            if not os.path.exists(path):
                with open(path, "w") as fd:
                    for obj in fn(*args, **kwargs):
                        fd.write(json.dumps(obj) + "\n")
            with open(path, "r") as fd:
                for line in fd:
                    yield json.loads(line)
        return wrapper
    return decorator


@cache(f"/home/{getpass.getuser()}/scratch/data/ja-clauses-hacked.jsonl")
def get_jas():
    return filter(
        lambda tup: tup[0],
        mapify(
            (spacify, clausify, labelify(1)),
            filter(fltr, map(tokenify, costep.sentences("english", "german")))
        )
    )


@cache(f"/home/{getpass.getuser()}/scratch/data/na-clauses-hacked.jsonl")
def get_nas():
    return filter(
        lambda tup: tup[0],
        mapify(
            (spacify, random_clausify, labelify(0)),
            filter(~fltr, map(tokenify, costep.sentences("english", "german")))
        )
    )


def get_data(ja_limit, na_limit):
    return chain(
        islice(get_jas(), ja_limit),
        islice(get_nas(), na_limit)
    )


def main(ctx: Context) -> None:
    ctx.parser.add_argument(
        "-o", "--outdir", default=f"/home/{getpass.getuser()}/scratch/data/"
    )
    ctx.parser.add_argument("-s", "--strategy", default="natural")
    ctx.parser.add_argument("-r", "--ratio", default=1.0, type=float)
    ctx.parser.add_argument("-d", "--seed", default=42, type=int)
    ctx.parser.add_argument("-j", "--ja-limit", type=int)
    ctx.parser.add_argument("-n", "--na-limit", type=int)
    args = ctx.parser.parse_args()
    # Set up logging.
    transformers.logging.disable_default_handler()
    transformers.logging.enable_propagation()
    transformers.logging.set_verbosity_error()
    # Prepare the data.
    random.seed(args.seed)
    # XXX: This is super inefficient on memory but imblearn doesn't know how
    #      to handle generators... Maybe I'll write my own version later.
    array = np.array(list(get_data(args.ja_limit, args.na_limit)))
    X, y = array[:, 0], array[:, 1]
    X = np.array(X).reshape(-1, 1)
    if args.strategy == "under":
        X_rslt, y_rslt = RandomUnderSampler(
            sampling_strategy=args.ratio,
            random_state=args.seed
        ).fit_resample(X, y)
    elif args.strategy == "over":
        X_rslt, y_rslt = RandomOverSampler(
            sampling_strategy=args.ratio,
            random_state=args.seed
        ).fit_resample(X, y)
    elif args.strategy == "natural":
        X_rslt, y_rslt = X, y
    else:
        ctx.parser.error(f"unknown strategy: {args.strategy}")
    # Log some basic stats.
    df = pd.DataFrame({"original": Counter(y), "resampled": Counter(y_rslt)}).T
    df = df.rename(columns={0: "nas", 1: "jas"})
    df["total"] = df.sum(axis=1)
    list(map(ctx.log.info, str(df).split("\n")))
    # Write the dataset to disk.
    outdir = f"clauses-corrected-{args.strategy}-{args.ratio}--{args.seed}"
    if args.strategy == "natural":
        outdir = f"clauses-corrected-{args.strategy}--{args.seed}"
    outdir = os.path.join(args.outdir, outdir)
    os.makedirs(outdir, exist_ok=True)
    # TODO: Make train/dev/test splits.
    to_dict = lambda snt, lbl: {"sentence": snt, "label": lbl}
    data = list(starmap(to_dict, zip(X_rslt.flatten(), y)))
    random.shuffle(data)
    train, dev, test = np.split(data, [int(len(data) * 0.7), int(len(data) * 0.8)])
    for name, itr in (("train", train), ("dev", dev), ("test", test)):
        path = os.path.join(outdir, f"{name}.jsonl")
        ctx.log.info("Writing %s", path)
        save_iter(map(json.dumps, itr), path)
    df = pd.DataFrame({
        "train": Counter(map(itemgetter("label"), train)),
        "dev": Counter(map(itemgetter("label"), dev)),
        "test": Counter(map(itemgetter("label"), test))
    }).T
    df = df.rename(columns={0: "nas", 1: "jas"})
    df["total"] = df.sum(axis=1)
    list(map(ctx.log.info, str(df).split("\n")))


if __name__ == "__main__":
    harness(main)
