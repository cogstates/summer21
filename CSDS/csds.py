# This is the Cognitive States Data Structure (CSDS) which represents information about writer cognitive state and
# the text from which we have gleaned this information.

# Basic information per instance of CognitiveStateFromText:
# - the text
# - the proposition in the text about which we record the cognitive state;
#     we represent this by the start and end of the syntactic headword of the
#     proposition
# - the belief value; the values used are corpus-specific (or experiment-specific),
#     for example CB, NCB, ROB, NA
# - the sentiment value; the values used are corpus-specific (or experiment-specific),
#     for example pos, neg

# Many more fields will be added as we go along.


class CognitiveStateFromText:
    text = ""  # sentence
    head_start = -1  # offset of start of head word of proposition
    head_end = -1  # offset of end of head word of proposition
    belief = ""  # belief value (values are corpus-specific)
    sentiment = ""  # sentiment value (values are corpus-specific)

    def __init__(self, this_text, this_head_start, this_head_end, this_belief):
        self.text = this_text
        self.head_start = this_head_start
        self.head_end = this_head_end
        self.belief = this_belief

    def get_info_short(self):
        return "<CSDS instance " + self.belief + "> "


class CSDS:
    instances = []
    corpus = ""

    def __init__(self, this_corpus):
        self.corpus = this_corpus


    def get_info_short(self):
        print("<CSDS from \"" + self.corpus + "\"; " + str(len(self.instances)) + " instances>")

    def get_info_long(self):
        print("<CSDS from \"" + self.corpus + "\"; " + str(len(self.instances)) + " instances:")
        for instance in self.instances:
            print("   " + instance.get_info_short())
        print(">")


