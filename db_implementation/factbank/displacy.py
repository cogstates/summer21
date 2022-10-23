import spacy
from spacy import displacy

# To either hold on tight or get out, as much of Asia goes into recession.

nlp = spacy.load("en_core_web_sm")
doc = nlp("People have predicted his demise so many times, and the US has tried to hasten it on several occasions.")
displacy.serve(doc, style="dep")