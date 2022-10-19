import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("He sold the property to five buyers and said he'd double his money.")
displacy.serve(doc, style="dep")