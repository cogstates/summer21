import spacy
from spacy import displacy

# To either hold on tight or get out, as much of Asia goes into recession.

nlp = spacy.load("en_core_web_sm")
doc = nlp('Mr. Stronach will direct an effort to reduce overhead and curb capital spending "until a more satisfactory level of profit is achieved and maintained," Magna said.')
displacy.serve(doc, style="dep")