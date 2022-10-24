import spacy

def get_head_span(head_token_offset_start, head_token_offset_end):
    fb_head_token = doc.char_span(head_token_offset_start, head_token_offset_end,
                                               alignment_mode='expand')[0]
    pred_head = current_sentence[head_token_offset_start:head_token_offset_end]
    print("FB HEAD: {}".format(pred_head))
    print("FB HEAD ANCESTORS & CHILDREN: {} {}".format(list(fb_head_token.ancestors), list(fb_head_token.children)))
    for token in list(fb_head_token.ancestors):
        print(token.text, token.pos_, token.n_lefts, list(token.lefts))

    if fb_head_token.dep_ == 'ROOT':
        syntactic_head_token = fb_head_token
    # elif fb_head_token.pos_ in ['PRON', 'PROPN', 'NOUN', 'VERB', 'AUX', 'NUM'] and fb_head_token.head.pos_ != 'ADP':
    #     syntactic_head_token = fb_head_token.head
    else:
        syntactic_head_token = None
        ancestors = list(fb_head_token.ancestors)
        if len(ancestors) == 1:
            syntactic_head_token = ancestors[0]
        else:
            for token in ancestors:
                if token.pos_ in ['PRON', 'PROPN', 'NOUN', 'VERB', 'AUX', 'NUM']:
                    syntactic_head_token = token
                    break

    # ancestors = list(syntactic_head_token.ancestors)
    # children = list(syntactic_head_token.children)
    print(syntactic_head_token.text)
    print(syntactic_head_token.left_edge.text)
    print(syntactic_head_token.right_edge.text)
    lefts = list(syntactic_head_token.lefts)
    print(lefts)
    if len(lefts) == 0:
        span_start = syntactic_head_token.left_edge.idx
    else:
        span_start = list(syntactic_head_token.lefts)[0].idx

    rights = list(syntactic_head_token.rights)
    print(rights)
    if len(rights) == 0:
        span_end = syntactic_head_token.right_edge.idx + len(syntactic_head_token.right_edge.text)
        # span_end_i = syntactic_head_token.right_edge.i
    else:
        span_end = rights[-1].idx + len(rights[-1].text)
        # span_end_i = rights[-1].i

    return span_start, span_end

nlp = spacy.load("en_core_web_sm")
current_sentence = "To either hold on tight or get out, as much of Asia goes into recession."
doc = nlp("To either hold on tight or get out, as much of Asia goes into recession.")
spans = get_head_span(62, 71)
print(doc.text[spans[0]:spans[1]])