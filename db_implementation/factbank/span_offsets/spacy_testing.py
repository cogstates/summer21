import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("On the other hand, it's turning out to be another very bad financial week for Asia.")
span = doc.char_span(24, 31)
head_token = span[0]
for child in head_token.children:
    print(child.text, child.dep_, child.head.text, child.head.pos_)


# for token in doc:
#     if token.head.text == 'turning':
#         print(token.text, token.dep_, token.head.text, token.head.pos_,
#               [child for child in token.children])
