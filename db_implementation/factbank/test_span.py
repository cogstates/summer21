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
    elif fb_head_token.pos_ in ['PRON', 'PROPN', 'NOUN', 'VERB', 'AUX', 'NUM'] and fb_head_token.head.pos_ != 'ADP':
        syntactic_head_token = fb_head_token.head
    else:
        syntactic_head_token = None
        ancestors = list(fb_head_token.ancestors)
        if len(ancestors) == 1:
            syntactic_head_token = ancestors[0]
        else:
            for token in ancestors:
                children = list(token.children)
                if token.pos_ in ['PRON', 'PROPN', 'NOUN', 'VERB', 'AUX', 'NUM']:
                    syntactic_head_token = token
                    break

    span_start = list(syntactic_head_token.lefts)[0].idx
    rights = list(syntactic_head_token.rights)
    span_end = rights[-1].idx + len(rights[-1].text)

    return span_start, span_end

nlp = spacy.load("en_core_web_sm")
current_sentence = "People have predicted his demise so many times, and the US has tried to hasten it on several occasions."
doc = nlp("People have predicted his demise so many times, and the US has tried to hasten it on several occasions.")
spans = get_head_span(26, 32)
print(doc.text[spans[0]:spans[1]])