# -*- coding: utf-8 -*-
import functools
import operator
import os
from typing import Any, Callable, Iterator, Optional, TypeVar, cast

import nltk  # type: ignore
import pycountry  # type: ignore

from ..core.functional import Filter, cache


_LIB_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
)
_ARCHIVE_PATH = os.path.join(_LIB_PATH, "archive")


_LANGS = {
    lang.alpha_2: lang.name.lower()
    for lang in pycountry.languages
    if hasattr(lang, "alpha_2")
}  # E.g. "de" -> "german"
_LANGS["el"] = "greek"  # Translates to Modern Greek.


# TODO: Add some logging here?
def download() -> None:
    """Download and upzip the CoStEP archive and nltk punkt."""
    import requests
    import shutil

    url = "http://pub.cl.uzh.ch/corpora/costep/costep_1.0.1.tar.gz"
    path = os.path.join(_ARCHIVE_PATH, os.path.basename(url))
    if os.path.exists(path):
        return
    # Download the corpus..
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    os.makedirs(_ARCHIVE_PATH, exist_ok=True)
    with open(path, "wb") as fd:
        fd.write(resp.raw.read())
    shutil.unpack_archive(path, _ARCHIVE_PATH)
    # Download punkt while we're at it.
    nltk.download("punkt")


@cache(os.path.join(_ARCHIVE_PATH, "sessions"))
def session(date: str) -> Iterator[dict[str, Any]]:
    """
    Read a session from the CoStEP archive.

    Note:
        This function will automatically download the corpus on first use. In addition,
        to avoid re-tokenizing the sentences it caches calls to a shelf..

    Args:
        date (str): The data in YYYY-MM-DD format.

    Yields:
        The next Speech block from the session.

    Examples:
        >>> next(costep.session("1996-04-15"))
            {'session': '1996-04-15',
             'chapter': '1',
             'turn': '1',
             'speaker': {'president': 'yes'},
             'texts': {'danish': ['Jeg erklærer Europa-Parlamentets... '],
              'german': ['Ich erkläre die am Donnerstag, den 28. März... '],
              'greek': ['Kηρύσσω την επανάληψη της συνόδου του...'],
              'english': ['I declare resumed the session of the European...'],
              'spanish': ['Declaro reanudado el período de sesiones del...'],
              'french': ['Je déclare reprise la session du Parlement européen...'],
              'italian': ['Dichiaro ripresa la sessione del Parlamento...'],
              'dutch': ['Ik verklaar de zitting van het Europees...'],
              'portuguese': ['Declaro reaberta a sessão do Parlamento...']}}
    """
    from xml.etree import ElementTree

    download()  # Download the corpus if we haven't already.
    root = ElementTree.parse(f"{_ARCHIVE_PATH}/sessions/{date}.xml")
    session, chapter, turn = None, None, None
    for tag in root.iter():
        if tag.tag == "session":
            session = tag.get("date")
        elif tag.tag == "chapter":
            chapter = tag.get("id")
        elif tag.tag == "turn":
            turn = tag.get("id")
        elif tag.tag == "speaker":
            assert not any([var is None for var in (session, chapter, turn)])
            spkr = {key: val for key, val in tag.attrib.items()}
            texts = {}
            for text in tag.findall(".//text"):
                if not text:
                    continue
                if text.get("language") in ("hu", "lt", "lv", "sk", "sl", "bg", "ro"):
                    continue  # TODO: Don't have the nltk model for it. Maybe just try-except here.
                lang = _LANGS[text.get("language")]
                paragraph = " ".join(
                    "".join(p.itertext()) for p in text.findall(".//p")
                )
                texts[lang] = nltk.tokenize.sent_tokenize(paragraph, language=lang)
            yield {
                "session": session,
                "chapter": chapter,
                "turn": turn,
                "speaker": spkr,
                "texts": texts,
            }
        else:
            pass  # Ignore


def dates() -> list[str]:
    """Return a sorted list of session dates."""
    from glob import glob

    return sorted(
        [
            os.path.splitext(os.path.basename(path))[0]
            for path in glob(f"{_ARCHIVE_PATH}/sessions/*")
        ]
    )


def speeches() -> Iterator[dict[str, Any]]:
    download()  # Download the corpus if we haven't already.
    for date in dates():
        for speech in session(date):
            yield speech


# fmt: off
def language(lang: str) -> Callable[[Filter], Filter]:
    """
    Creates a function that when given a filter will modify the input
    to apply to a particular language from the corpus.

    Args:
        lang (str): The language for the filter (E.g."english")

    Returns:
        A function that will modify a filter to apply to that language.

    Examples:
        >>> german = language("german")
        >>> next(costep.search(german(contains("ja")), ("german", "english")))
        {'german': 'Wir haben ja mit Ihnen einen Experten, der ohnehin mit diesen...',
         'english': 'After all, we have in you an expert who is in any case closely...',
         'meta': {'session': '1996-04-15',
          'chapter': '3',
          'turn': '11',
          'speaker': {'president': 'yes'}}}
    """
    def decorator(fn: Filter) -> Filter:
        def fltr(item: dict[str, str]) -> bool:
            return fn(item[lang].lower().split())
        return Filter(fltr)
    return decorator
# fmt: on


def contains(phrase: str) -> Filter:
    """
    Creates a filter that will search for sentences containing the given phrase.

    Args:
        phrase(str): A space delimited phrase to search for.

    Returns:
        Filter: A filter which searches sentences for that phrase.
    """
    phr = tuple(phrase.lower().split())

    def fltr(sent: list[str]) -> bool:
        return phr in nltk.FreqDist(nltk.ngrams(sent, len(phr)))

    return Filter(fltr)


def starts_with(phrase: str) -> Filter:
    """
    Creates a filter that will search for sentences starting with the given phrase.

    Args:
        phrase(str): A space delimited phrase to search for.

    Returns:
        Filter: A filter which searches for sentences starting with that phrase.
    """
    phr = phrase.lower().split()

    def fltr(sent: list[str]) -> bool:
        return sent[: len(phr)] == phr

    return Filter(fltr)


def every(*args: Filter) -> Filter:
    """
    Logically and's a bunch of filters.

    Args:
        *args (list[Filter]): List of filters to logically and.

    Returns:
        Filter: A filter that logically and's the input filters.

    Examples:
        >>> german = language("german")
        >>> next(costep.search(german(every(contains("ja"), contains("sagen"))), ("german", "english")))
        {'german': 'Daß wir ja zur Zollunion sagen müssen, weil so die islamische Gefahr gebannt...',
         'english': 'That we should approve the customs union as a means to curbing the threat...',
         'meta': {'session': '1996-06-20',
          'chapter': '6',
          'turn': '63',
          'speaker': {'president': 'no',
           'name': 'Kaklamanis',
           'language': 'el',
           'forename': 'Nikitas',
           'surname': 'Kaklamanis',
           'country': 'GR',
           'group': 'UPE',
           'id': '1185'}}}
    """
    return functools.reduce(operator.and_, args)


def some(*args: Filter) -> Filter:
    """
    Logically or's a bunch of filters.

    Args:
        *args (list[Filter]): List of filters to logically or.

    Returns:
        Filter: A filter that logically or's the input filters.

    Examples:
        >>> german = language("german")
        >>> next(costep.search(german(~every(~contains("ja"), ~contains("sagen"))), ("german", "english")))
        {'german': 'Daß wir ja zur Zollunion sagen müssen, weil so die islamische Gefahr gebannt...',
         'english': 'That we should approve the customs union as a means to curbing the threat...',
         'meta': {'session': '1996-06-20',
          'chapter': '6',
          'turn': '63',
          'speaker': {'president': 'no',
           'name': 'Kaklamanis',
           'language': 'el',
           'forename': 'Nikitas',
           'surname': 'Kaklamanis',
           'country': 'GR',
           'group': 'UPE',
           'id': '1185'}}}
    """
    return functools.reduce(operator.or_, args)


def search(fltr: Filter, langs: list[str]) -> Iterator[Any]:
    """
    Searches CoStEP sentences using the given filter and returns parallel text for
    the requested languages.

    Args:
        fltr (Filter): A filter to search the corpus with.
        langs (list[str]): The languages to return data for.

    Yields:
        Data for the next sentence matching the given filter.

    Examples:
        >>> german = language("german")
        >>> english = language("english")
        >>> next(costep.search(german(contains("ja")) & english(contains("that")), ("german", "english")))
        {'german': 'Herr Verfasser der Stellungnahme, das ist ja ein amüsanter Zufall, ...',
         'english': 'Mr Soulier, it is amusing that the version in which there is a...',
         'meta': {'session': '1996-04-15',
          'chapter': '6',
          'turn': '12',
          'speaker': {'president': 'yes'}}}
    """
    for speech in speeches():
        if not all([lang in speech["texts"] for lang in langs]):
            continue  # Parse error? File didn't support that language?
        if not len({len(speech["texts"][lang]) for lang in langs}) == 1:
            continue  # Sentences didn't line up.
        for idx in range(len(speech["texts"][langs[0]])):
            sent = {lang: speech["texts"][lang][idx] for lang in langs}
            sent["meta"] = {
                attr: speech[attr] for attr in ("session", "chapter", "turn")
            }
            sent["meta"]["speaker"] = speech["speaker"]
            if fltr(sent):
                yield sent
