import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Ah, we just saw an explosion up ahead of us here about sixteen thousand feet or something like that.")
displacy.serve(doc, style="dep")