sentence = "In nineteen ninety-one Charles Keating was convicted in state court of helping to defraud thousands of investors who bought high risk junk bonds sold by Keating's employees at Lincoln savings and loan."
offset_start = 145
offset_end = 149
head = sentence[offset_start:offset_end]

def final_check(head, sentence, offset_start, offset_end):
    print("\n\nHEAD SHOULD BE: " + head + "\n\n")
    print("\n\nWITH ASTERISKS...\n\n")
    print(sentence[:offset_start] + "* " + head + " *" + sentence[offset_end:] + "\n")
    print("\n\nORIGINAL SENTENCE FOR VERIFICATION...\n\n")
    print(sentence)

def check_length(chunk):
    print("Starting index should be: " + str(len(chunk) - 1))

# final_check(head, sentence, offset_start, offset_end)

def populate_fixed_errors():
    f = open('fb_fixed_errors.txt', 'r')
    data = f.readline().split(",")
    file = data[0]
    file_sentence_id = int(data[1])
    sentence = f.readline().replace('\n','')
    head = f.readline().replace('\n','')
    rel_source_text = f.readline().replace('\n','')
    fact_value = f.readline().replace('\n','')
    offsets = f.readline().split(',')
    offset_start = offsets[0]
    offset_end = offsets[1]

    print('FILE: {0}'.format(file))
    print('FILE_SENTENCE_ID: {0}'.format(file_sentence_id))
    print('SENTENCE: {0}'.format(sentence))
    print('HEAD: {0}'.format(head))
    print('REL_SOURCE_TEXT: {0}'.format(rel_source_text))
    print('FACT_VALUE: {0}'.format(fact_value))
    print('OFFSET_START: {0}'.format(offset_start))
    print('OFFSET_END: {0}'.format(offset_end))

# populate_fixed_errors()
final_check(head, sentence, offset_start, offset_end)
# check_length("Well may we say God save the Queen,' for nothing will s")
