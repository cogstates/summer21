import spacy

def get_head_span(head_token_offset_start, head_token_offset_end):
    fb_head_token = doc.char_span(head_token_offset_start, head_token_offset_end,
                                               alignment_mode='expand')[0]
    pred_head = current_sentence[head_token_offset_start:head_token_offset_end]
    print("FB HEAD: {}".format(pred_head))
    print("FB HEAD ANCESTORS & CHILDREN: {} {}".format(list(fb_head_token.ancestors), list(fb_head_token.children)))

    if fb_head_token.dep_ == 'ROOT':
        syntactic_head_token = fb_head_token
    elif fb_head_token.pos_ in ['PRON', 'PROPN', 'NOUN', 'VERB', 'AUX', 'NUM']:
        syntactic_head_token = fb_head_token.head
    else:
        syntactic_head_token = None
        ancestors = list(fb_head_token.ancestors)
        for token in ancestors:
            if token.pos_ in ['PRON', 'PROPN', 'NOUN', 'VERB', 'AUX', 'NUM']:
                syntactic_head_token = token
                break

    span_start, span_end = None, None
    ancestors = list(syntactic_head_token.ancestors)
    children = list(syntactic_head_token.children)
    print("FIRST PASS ANCESTORS: {}".format(ancestors))
    print("FIRST PASS CHILDREN: {}".format(children))
    while len(ancestors) == 1 and len(children) == 1:
        span_end = syntactic_head_token.right_edge.idx + len(syntactic_head_token.right_edge.text)
        syntactic_head_token = syntactic_head_token.head

        ancestors = list(syntactic_head_token.ancestors)
        children = list(syntactic_head_token.children)
    span_start = syntactic_head_token.left_edge.idx
    if span_end is None:
        span_end = syntactic_head_token.right_edge.idx + len(syntactic_head_token.right_edge.text)

    first_span = current_sentence[span_start:span_end]
    if pred_head not in first_span:
        ancestors = list(syntactic_head_token.ancestors)
        children = list(syntactic_head_token.children)
        children_text = [child.text for child in children]

        if ',' in children_text and children_text.index(',') > len(children_text) // 2:
            children = children[:children_text.index(',') + 1]

        if children[-1].pos_ == 'PUNCT':
            children = children[:-1]
        span_start = syntactic_head_token.left_edge.idx
        span_end = children[-1].idx + len(children[-1].text)

    return span_start, span_end

nlp = spacy.load("en_core_web_sm")
current_sentence = "To either hold on tight or get out, as much of Asia goes into recession."
doc = nlp("To either hold on tight or get out, as much of Asia goes into recession.")
spans = get_head_span(62, 71)
print(doc.text[spans[0]:spans[1]])