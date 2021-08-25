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

from itertools import chain


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
    """
    Holds a collection of CSDS objects for a single corpus, each of which represents
    a single example in the corpus.
    Each example consists of a sentence in the corpus, together with a label
    annotating a word or phrase in the sentence, which is called the head.
    A single collection represents an entire corpus.
    Maintains separate lists of CSDS objects whose labels correspond
    to actual annotations and to default pseudo-annotations (for un-annotated
    tokens), respectively.  The pseudo-annotation appears as the 'O' label here.
    The client code determines whether to populate the second list.
    """
    # List of examples from the corpus that were originally annotated with a real label.
    labeled_instances = []

    # List of examples consisting of non-annotated words with the "O" pseudo-label.
    o_instances = []

    # Name of the corpus from which the examples in this collection were drawn.
    # This collection represents a single corpus.
    corpus = ""

    def __init__(self, this_corpus):
        """
        Stores the name of the corpus from which the examples are drawn.
        :param this_corpus:
        """
        self.corpus = this_corpus

    def add_labeled_instance(self, new_instance):
        """
        Adds a single CSDS object to the collection for an actual annotation.
        :param new_instance: The CSDS object representing the example.
        :return: None.
        """
        self.labeled_instances.append(new_instance)

    def add_o_instance(self, new_instance):
        """
        Adds a single CSDS object to the collection for an un-annotated word.
        :param new_instance: The CSDS object representing the example.
        :return: None.
        """
        self.o_instances.append(new_instance)

    def add_list_of_labeled_instances(self, list_of_new_instances):
        """
        Adds a list of CSDS objects to this collection, where each
        CSDS object corresponds to an actual annotation in the corpus.
        :param list_of_new_instances: List of CSDS objects with a label based on an annotation.
        :return: None.
        """
        self.labeled_instances.extend(list_of_new_instances)

    def add_list_of_o_instances(self, list_of_new_instances):
        """
        Adds a list of CSDS objects to this collection, where each
        CSDS object corresponds to word in the corpus that has not been annotated.
        :param list_of_new_instances: List of CSDS objects with the 'O' label.
        :return: None.
        """
        self.o_instances.extend(list_of_new_instances)

    def get_all_instances(self):
        """
        Returns two lists of CSDS objects:
        1. The first corresponding to actual annotations in the corpus
        2. The second corresponding to non-annotated words in the corpus.
        :return: A pair of lists of CSDS objects.
        """
        return self.labeled_instances, self.o_instances

    def get_next_instance(self):
        """
        Provides for iteration over all CSDS objects in the collection.
        :return: An iterator that includes all internal lists of CSDS objects.
        """
        return chain(self.labeled_instances, self.o_instances)

    def get_next_labeled_instance(self):
        """
        Provides for iteration over only those CSDS objects in the collection
        that correspond to annotations in the corpus.
        :return: An iterator that includes only the list of labeled instances.
        """
        for instance in self.labeled_instances:
            yield instance

    def get_next_o_instance(self):
        """
        Provides for iteration over only those CSDS objects in the collection
        that correspond to words not annotated in the corpus.
        :return: An iterator that includes only the list of instances labeled 'O.'
        """
        for instance in self.o_instances:
            yield instance

    def get_num_labeled_instances(self):
        """
        Gets the number of CSDS objects in this collection corresponding
        to actual annotations in the corpus.
        :return:  An integer, the count of labeled CSDS objects.
        """
        return len(self.labeled_instances)

    def get_o_instances_length(self):
        """
            Gets the number of CSDS objects in this collection corresponding
            to un-annotated instances of words in the corpus.
            :return:  An integer, the count of 'O'-labeled CSDS objects.
            """
        return len(self.o_instances)

    def get_info_short(self):
        """
        Gets a brief string representation of this collection.
        :return: A string containing essential details about this collection.
        """
        return (
                f"<CSDS collection from \"{self.corpus}\": {str(len(self.labeled_instances))} " 
                f"labeled_instances>"
        )

    def get_info_long(self):
        """
        Gets a detailed string representation of the collection.
        :return: A string including representations of each labeled instance in the collection.
        """
        message = (
            f"<CSDS collection from \"{self.corpus}\": {str(len(self.labeled_instances))} "
            f"labeled_instances:\n"
        )
        for instance in self.labeled_instances:
            message += f"   {instance.get_info_short()}\n"
        message += ">\n"
        return message
