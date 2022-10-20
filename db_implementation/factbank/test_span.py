import spacy

def get_head_span(head_token_offset_start, head_token_offset_end):
    fb_head_token = doc.char_span(head_token_offset_start, head_token_offset_end,
                                               alignment_mode='expand')[0]

    if fb_head_token.dep_ == 'ROOT':
        syntactic_head_token = fb_head_token
    elif fb_head_token.pos_ in ['PRON', 'PROPN', 'NOUN', 'VERB', 'AUX', 'NUM']:
        syntactic_head_token = fb_head_token.head
    else:
        syntactic_head_token = None
        ancestors = list(fb_head_token.ancestors)
        # print(ancestors)
        for token in ancestors:
            if token.pos_ in ['PRON', 'PROPN', 'NOUN', 'VERB', 'AUX', 'NUM']:
                syntactic_head_token = token
                break

    ancestors = list(syntactic_head_token.ancestors)
    children = list(syntactic_head_token.children)
    children_text = [child.text for child in children]
    print(syntactic_head_token.text, ancestors, children)
    one_ancestor_child = False
    right_edge = None
    if len(ancestors) == 1 and len(children) == 1:
        print('One ancestor child')
        one_ancestor_child = True
        right_edge = syntactic_head_token.right_edge.idx + len(syntactic_head_token.right_edge.text)
        syntactic_head_token = syntactic_head_token.head
    elif ',' in children_text and children_text.index(',') > len(children_text) // 2:
        children = children[:children_text.index(',') + 1]

    # if children[-1].pos_ == 'PUNCT':
    #     children = children[:-1]
    span_start = syntactic_head_token.left_edge.idx
    if one_ancestor_child:
        span_end = right_edge
    else:
        span_end = children[-1].idx + len(children[-1].text)

    return span_start, span_end

nlp = spacy.load("en_core_web_sm")
doc = nlp("People have predicted his demise so many times, and the US has tried to hasten it on several occasions.")
spans = get_head_span(19, 28)
print(doc.text[spans[0]:spans[1]])