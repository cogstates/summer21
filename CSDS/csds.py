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
        return "<CSDS instance " + self.text + " " + self.belief + "> "


class CSDS:
    instances = []
    corpus = ""

    def __init__(self, this_corpus):
        self.corpus = this_corpus

    # add a single new instance
    def add_instance(self, new_instance):
        self.instances.append(new_instance)

    # add a list of new instances
    def add_list_of_instances(self, list_of_new_instances):
        self.instances.extend(list_of_new_instances)

    # return instances as list
    def get_all_instances(self):
        return self.instances

    # generator: return the next instance in an iteration over the instances
    def get_next_instance(self):
        for instance in self.instances:
            yield instance

    def get_info_short(self):
        return "<CSDS from \"" + self.corpus + "\"; " + str(len(self.instances)) + " instances>"

    def get_info_long(self):
        message = "<CSDS from \"" + self.corpus + "\"; " + str(len(self.instances)) + " instances:\n"
        for instance in self.instances:
            message += "   " + instance.get_info_short() + "\n"
        message += ">\n"
        return message
