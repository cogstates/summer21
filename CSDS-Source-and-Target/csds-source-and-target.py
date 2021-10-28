from dataclasses import dataclass


# Cognitive States Data Structure (CSDS): Represents information about
# the writer's cognitive state and the text from which we have gleaned
# this information.



# Basic information per CSDS instance:
# - the text (sentence) we are processing
# - the passage in the text about which we record the cognitive state;
#     we represent this by the start and end of the syntactic headword of the
#     proposition
# - the belief value; the values used are corpus-specific (or experiment-specific),
#     for example CB, NCB, ROB, NA
# - the sentiment value; the values used are corpus-specific (or experiment-specific),
#     for example pos, neg

# Many more fields will be added as we go along.


@dataclass
class TextReferent:
    """
    A text referent is an object in a different representation system (not text-bound) in which
    different things in the world correspond to different objects, but each real-world thing
    has only one object which represents it.  A "thing" can be an entity (concrete or abstract) or
    an eventuality (an event or a state), or perhaps other things.
    """

    text_passages: list  # list of text passages
    name: str = ""  # name from underlying annotation, such as "e23"
    type: str = ""  # types will vary by corpus, but could be "event" and "entity"
    subtype: str = ""  # some types may have subtypes *(ntities can be person, org, etc), corpus-specific

    # Add a passage to the list of passages associated with this referent
    def add_passage(self, text_passage):
        self.text_passages.append(text_passage)


@dataclass
class TextPassage:
    """
    A text passage is a span in a sentence which refers to the source or target of a cognitive state.
    A corpus can annotate a head or a span or both; machine learning can require a head or a span or
    both; if one is missing, we can obtain it using parsing.

    A text passage can also be the author with no textual manifestation in this sentence, in which
    case all instance variables related to head and span remain -1, but the is_author flag is set to
    true.
    """

    referent: TextReferent  # pointer to an object of type TextReferent

    sentence: str = ""  # sentence in which the annotated head occurs
    sentence_id: str = ""  # sentence id of the sentence in which the annotated passage occurs
    document_id: str = ""  # document id of the sentence in which the annotated head occurs
    head_start: int = -1  # offset within sentence of start of head word of passage
    head_end: int = -1  # offset of end of head word of passage
    head: str = ""  # head word of passage (can be determined from text, head_start, head_end)
    span_start: int = -1  # offset within sentence of start of test passage
    span_end: int = -1  # offset of end of passage
    span: str = ""  # text passage (can be determined from text, span_start, span_end)
    is_author: bool = True  # Is this the author (whether with or without textual manifestation)

    # tailor-made initialization functions

    # head-only text passage
    def __init__(self, this_text, this_head_start, this_head_end, this_head=""):
        self.text = this_text
        self.head_start = this_head_start
        self.head_end = this_head_end
        self.head = this_head


@dataclass
class CSDS:
    """
    Cognitive States Data Structure (CSDS): Represents information about 
    cognitive agents' cognitive state and the text from which we have gleaned
    this information.
    """

    source: list  # list of text passages = nested attribution; first element always author
    target: TextPassage = None  # what the attitude is about
    belief = ""  # belief value (values are corpus-specific)
    sentiment = ""  # sentiment value (values are corpus-specific)

    # tailor-made inits

    # init for target-only case (we do not even instantiate the source, assume it is not
    # needed for machine learning)
    def __init__(self, this_text, this_head_start, this_head_end, this_belief, this_head=""):

        my_text_passage = TextPassage(this_text, this_head_start, this_head_end, this_head)
        self.target = my_text_passage
        self.belief = this_belief

    # @TODO expand this to check if source and if yes, include in printout
    def get_info_short(self):
        if self.target is None:
            return "<CSDS instance  (" + self.belief + ") (no target)> "
        else:
            return "<CSDS instance (" + str(self.target.head_start) + ") (" + self.target.text + ") (" + \
                   self.target.head + ") (" + self.belief + ")> "




@dataclass
class CSDSCollection:
    instances: list      # list of instances of cognitive states
    corpus: str = ""     # corpus from which they are drawn

    # I have not changed the code below this line

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


#########################################################
## General functions
#########################################################

# Create both a text passage and a text referent, link them up properly,
# and return the text passage
def create_headed_text_passage_and_referent (sentence, head_start, head_end, head, referent_name):
    my_text_referent = create_named_text_referent(referent_name)
    my_text_passage = create_headed_text_passage(head, head_end, head_start, sentence)
    mutually_link(my_text_passage, my_text_referent)
    return my_text_passage


# Create a text referent with a name
def create_named_text_referent(referent_name):
    my_text_referent = TextReferent()
    my_text_referent.name = referent_name
    return my_text_referent


# Create a text passage with head info
def create_headed_text_passage(head, head_end, head_start, sentence):
    my_text_passage = TextPassage()
    my_text_passage.head_start = head_start
    my_text_passage.head_end = head_end
    my_text_passage.head = head
    my_text_passage.sentence = sentence
    return my_text_passage


# Add mutual links between a text passage and a text referent
def mutually_link(my_text_passage, my_text_referent):
    my_text_referent.text_passages.append(my_text_passage)
    my_text_passage.referent = my_text_referent