sentence = \
    "In Saudi Arabia, today is the eighth day of Desert Shield,'' the operation designed to confront an estimated 200,000 Iraqi soldiers now in Kuwait."
offset_start = 136
offset_end = 138
head = sentence[offset_start:offset_end]

def final_check(head, sentence, offset_start, offset_end):
    print("\n\nHEAD SHOULD BE: " + head + "\n\n")
    print("\n\nWITH ASTERISKS...\n\n")
    print(sentence[:offset_start] + "* " + head + " *" + sentence[offset_end:] + "\n")
    print("\n\nORIGINAL SENTENCE FOR VERIFICATION...\n\n")
    print(sentence)

def check_length(chunk):
    print("Starting index should be: " + str(len(chunk) - 1))

final_check(head, sentence, offset_start, offset_end)
