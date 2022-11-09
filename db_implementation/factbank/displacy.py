import spacy
from spacy import displacy

# To either hold on tight or get out, as much of Asia goes into recession.
# Mr. Stronach will direct an effort to reduce overhead and curb capital spending "until a more satisfactory level of profit is achieved and maintained," Magna said.

nlp = spacy.load("en_core_web_lg")
doc = nlp('Mr. Stronach will direct an effort to reduce overhead and curb capital spending "until a more satisfactory level of profit is achieved and maintained," Magna said.')

for token in doc:
    print(token.text)
    sub_tree = list(token.subtree)
    print(sub_tree)
    for sub_token in sub_tree:
        print(sub_token.text)
        print(list(sub_token.ancestors))
    print("--------------------------\n\n")

displacy.serve(doc, style="dep")

