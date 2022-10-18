import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("She estimates her properties, worth a hundred thirty million dollars in October, "
          "are worth only half that now.")
displacy.serve(doc, style="dep", options={"compact":True})