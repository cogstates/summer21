import spacy

# if target is not noun or verb, go up the subtree, check noun then verb, then stop

nlp = spacy.load('en_core_web_sm')
doc = nlp("John says that Mary is coming to dinner.")
for token in doc:
    print(token.text, token.dep_, token.head.text, token.pos_,
          [child for child in token.children])


# raw_sentence = "On the other hand, it's turning out to be another very bad financial week for Asia."
# span = doc.char_span(24, 31)
# head_token = span[0]
# head_children = [child for child in head_token.children]
# for child in head_token.children:
#     print(child.text, child.dep_, child.head.text, child.head.pos_, child.idx)
# span_start = min(child.idx for child in head_token.children)
# span_end_token = [child for child in head_token.children if child.idx == \
#                   max(child.idx for child in head_token.children if child.dep_ != 'punct')][0]
# span_end = span_end_token.idx + len(span_end_token.text)
#
# print(raw_sentence[span_start:span_end])

# def get_head_span(raw_sentence, head_token_offset_start, head_token_offset_end):
#     doc = nlp(raw_sentence)
#     span = doc.char_span(head_token_offset_start, head_token_offset_end)
#     head_token = span[0]
#     span_start = min(child.idx for child in head_token.children)
#     span_end_token = [child for child in head_token.children if child.idx == \
#                       max(child.idx for child in head_token.children if child.dep_ != 'punct')][0]
#     span_end = span_end_token.idx + len(span_end_token.text)
#
#     return (span_start, span_end)
#
# test_sentences = [["The financial assistance from the World Bank and the International Monetary Fund are not helping.", 14, 24],\
# ["In the last twenty four hours, the value of the Indonesian stock market has fallen by twelve percent.", 76, 82]]
#
# # for test_sentence in test_sentences:
# #     head_span = get_head_span(test_sentence[0], test_sentence[1], test_sentence[2])
# #     print(test_sentence[0])
# #     print("HEAD: {}".format(test_sentence[0][test_sentence[1]:test_sentence[2]]))
# #     print(test_sentence[0][head_span[0]:head_span[1]])
# #     print('-----------------------')
#
# doc = nlp("Kopp's stepmother, who married Kopp's father when Kopp was in his 30s, said Thursday from her home in Irving, Texas: \"I would like to see him come forward and clear his name if he's not guilty, and if he's guilty, to contact a priest and make his amends with society, face what he did.")
# for token in doc:
#     print(token.text, token.dep_, token.head.text, token.head.pos_, token.idx,
#           [child for child in token.children])
