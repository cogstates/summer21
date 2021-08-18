# Cognitive States Data Structure (CSDS): Represents information about
# the writer's cognitive state and the text from which we have gleaned
# this information.
# Basic information per CSDS instance:
# - the text
# - the proposition in the text about which we record the cognitive state;
#     we represent this by the start and end of the syntactic headword of the
#     proposition
# - the belief value; the values used are corpus-specific (or experiment-specific),
#     for example CB, NCB, ROB, NA
# - the sentiment value; the values used are corpus-specific (or experiment-specific),
#     for example pos, neg

# Many more fields will be added as we go along.


class CSDS:
    """
    Cognitive States Data Structure (CSDS): Represents information about 
    the writer's cognitive state and the text from which we have gleaned 
    this information.
    """
    doc_id = -1  # unique index of origin document within corpus
    sentence_id = -1  # index of sentence within document
    text = ""  # sentence in which the annotated head occurs
    head_start = -1  # offset within sentence of start of head word of proposition
    head_end = -1  # offset of end of head word of proposition
    head = ""  # target of annotation within sentence
    belief = ""  # belief value (values are corpus-specific)
    sentiment = ""  # sentiment value (values are corpus-specific)

    def __init__(
            self, this_text, this_head_start, this_head_end, this_belief, this_head="",
            this_doc_id=-1, this_sentence_id=-1
    ):
        self.doc_id = this_doc_id
        self.sentence_id = this_sentence_id
        self.text = this_text
        self.head_start = this_head_start
        self.head_end = this_head_end
        self.belief = this_belief
        self.head = this_head

    def get_info_short(self):
        return (
            f"<CSDS Doc: {self.doc_id} Sentence: {self.sentence_id} Head: {self.head_start} "
            f"Text {self.text} Head: {self.head} Belief: {self.belief}"
        )

    def get_marked_text(self):
        # puts stars around annotated snippet
        new_sentence = self.text[0:self.head_start] + "* " + self.text[self.head_start:self.head_end] + \
                       " *" + self.text[self.head_end: len(self.text)]
        return new_sentence

    def get_belief(self):
        return self.belief

    def get_doc_id(self):
        return self.doc_id

class CSDSCollection:
    labeled_instances = []
    o_instances = []
    corpus = ""

    def __init__(self, this_corpus):
        self.corpus = this_corpus

    # add a single new instance
    def add_instance(self, new_instance, instance_type='labeled'):
        if instance_type == 'o':
            self.o_instances.append(new_instance)
        else:
            self.labeled_instances.append(new_instance)

    # add a list of new labeled_instances
    def add_list_of_instances(self, list_of_new_instances, instance_type='labeled'):
        if instance_type == 'o':
            self.o_instances.extend(list_of_new_instances)
        else:
            self.labeled_instances.extend(list_of_new_instances)

    # return labeled_instances as list
    def get_all_instances(self):
        return self.labeled_instances, self.o_instances

    # generator: return the next instance in an iteration over the labeled_instances
    def get_next_instance(self, instance_type='labeled'):
        if instance_type == 'o':
            instances = self.o_instances
        else:
            instances = self.labeled_instances
        for instance in instances:
            yield instance

    def get_labeled_instances_length(self):
        return len(self.labeled_instances)

    def get_o_instances_length(self):
        return len(self.o_instances)

    def get_info_short(self):
        return "<CSDS from \"" + self.corpus + "\"; " + str(len(self.labeled_instances)) + " labeled_instances>"

    def get_info_long(self):
        message = "<CSDS from \"" + self.corpus + "\"; " + str(len(self.labeled_instances)) + " labeled_instances:\n"
        for instance in self.labeled_instances:
            message += "   " + instance.get_info_short() + "\n"
        message += ">\n"
        return message
