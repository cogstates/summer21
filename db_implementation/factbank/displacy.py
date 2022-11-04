import spacy
from spacy import displacy

# To either hold on tight or get out, as much of Asia goes into recession.
# Mr. Stronach will direct an effort to reduce overhead and curb capital spending "until a more satisfactory level of profit is achieved and maintained," Magna said.

nlp = spacy.load("en_core_web_sm")
doc = nlp("So when Wong Kwan spent seventy million dollars for this house, he thought it was a great deal.")

# for token in doc:
#     print(token.text, token.dep_)


displacy.serve(doc, style="dep")

