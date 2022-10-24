import spacy
from spacy import displacy

# To either hold on tight or get out, as much of Asia goes into recession.

nlp = spacy.load("en_core_web_sm")
doc = nlp("Hotels are only thirty percent full.")
displacy.serve(doc, style="dep")