import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("To either hold on tight or get out, as much of Asia goes into recession.")
displacy.serve(doc, style="dep")