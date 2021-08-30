import glob
import xml.etree.ElementTree as et
import re
from CSDS.csds import CSDS, CSDSCollection
from nltk.tokenize import SpaceTokenizer


class XMLCorpusToCSDSCollection:
    """
    Class to create a Cognitive State Data Structure (CSDS) collection
    corresponding to a corpus consisting of XML files with annotations
    on text targets (heads) following the GATE format.
    """

    corpus_name = ""

    # Can be a relative or absolute path to the directory holding
    # the XML files of the corpus:
    corpus_directory = ""

    # This will be the CSDSCollection object, a collection
    # of data structures, each of which represents a single
    # example pairing some element of a sentence with a label.
    csds_collection = None

    # A unique, sequential integer identifying the document
    # within the corpus being processed on a call to add_file.
    # In normal usage, create_and_get_collection iterates
    # through all the files of the corpus and calls add_file
    # on each file.  doc_id represents the sequence number of this iteration.
    doc_id = -1

    # Map from node ID number to the sentences within which the node
    # element occurs. Node id numbers are unique within a given document.
    nodes_to_sentences = {}

    # Map from node ID number to the text snippet with which it is
    # associated. This may be an annotation head, but the map includes
    # text following a head end node as well.
    nodes_to_targets = {}

    # Map from node ID to a pair of indices giving the locations within
    # the sentence of the start and one past the end of the text immediately
    # following the node element in the XML file and associated with the node.
    nodes_to_offsets = {}

    # List of sentences within the current document, kept in sequence.
    sentences = []

    # Map from each sentence to a list of pairs of indices within the
    # sentence, each of which pair describes an annotation target (head).
    sentence_to_annotation_offsets = {}

    def __init__(self, corpus_name, corpus_directory):
        self.corpus_name = corpus_name
        self.corpus_directory = corpus_directory
        self.csds_collection = CSDSCollection(self.corpus_name)

    def update_nodes_dictionaries(self, tree):
        """
        Gathers the data for the current document concerning
        node elements in the document text, as well as the
        sentences constituting the text.
        :param tree: Root of the XML parse tree.
        :return: None.
        """
        text_with_nodes = tree.find('TextWithNodes')
        nodes_in_sentence = []
        sentence = ""
        sentence_id = 0
        if text_with_nodes.text is not None:
            sentence += text_with_nodes.text
        sentence_length_so_far = len(sentence)
        self.nodes_to_targets[0] = text_with_nodes.text
        for node in text_with_nodes.findall('Node'):
            text = node.tail
            node_id = node.attrib['id']
            if text is None:
                continue
            self.nodes_to_targets[node_id] = text
            nodes_in_sentence.append(node_id)
            self.nodes_to_offsets[node_id] = sentence_length_so_far
            if '\n' in text:
                parts = text.split('\n')
                sentence += parts[0]
                self.sentence_to_annotation_offsets[sentence_id] = []
                self.sentences.append(sentence)
                for node_in_sentence in nodes_in_sentence:
                    self.nodes_to_sentences[node_in_sentence] = sentence_id
                nodes_in_sentence.clear()
                sentence = parts[-1]
                sentence_length_so_far = len(sentence)
                sentence_id += 1
            else:
                sentence += text
                sentence_length_so_far += len(text)

    def add_file_to_csds_collection(self, tree, xml_file):
        """
        Processes a single XML file after initial parsing, constructs
        a CSDS object for each annotation, and adds the CSDS object to the
        growing collection.
        :param tree: Root of the XML parse tree.
        :param xml_file: String, name of the current XML file.
        :return: None.
        """
        self.doc_id += 1
        annotation_sets = tree.findall('AnnotationSet')
        for annotation_set in annotation_sets:
            for annotation in annotation_set:
                if annotation.attrib['Type'] == 'paragraph':
                    continue
                node_id = annotation.attrib['StartNode']
                head_start = self.nodes_to_offsets[node_id]
                target_length = len(self.nodes_to_targets[node_id])
                length_check = int(annotation.attrib['EndNode']) - int(node_id)
                if length_check != target_length:
                    print(f'File: {xml_file} - Node: {node_id} has an end marking mismatch.')
                head_end = head_start + target_length
                annotation_type = annotation.attrib['Type']
                annotation_type = re.sub(r'\s*future$', '', annotation_type, flags=re.I)
                sentence_id = self.nodes_to_sentences[annotation.attrib['StartNode']]
                cog_state = CSDS(
                    self.sentences[sentence_id],
                    head_start,
                    head_end,
                    annotation_type,
                    self.nodes_to_targets[annotation.attrib['StartNode']],
                    self.doc_id,
                    sentence_id
                )
                self.csds_collection.add_labeled_instance(cog_state)
                self.sentence_to_annotation_offsets[sentence_id].append((head_start, head_end))

    def add_file(self, xml_file):
        """
        Parses the current file, processes the corresponding data, and
        cleans up the data structures for the next document.
        :param xml_file: String, name of the current XML file.
        :return: None.
        """
        tree = et.parse(xml_file)
        self.update_nodes_dictionaries(tree)
        self.add_file_to_csds_collection(tree, xml_file)
        self.nodes_to_sentences.clear()
        self.nodes_to_targets.clear()
        self.nodes_to_offsets.clear()
        self.sentences.clear()

    def create_and_get_collection(self):
        for file in glob.glob(self.corpus_directory + '/*.xml'):
            self.add_file(file)
        return self.csds_collection


class XMLCorpusToCSDSCollectionWithOLabels(XMLCorpusToCSDSCollection):
    """
    Class to create a Cognitive State Data Structure (CSDS) collection
    corresponding to a corpus consisting of XML files with annotations
    on text targets (heads) following the GATE format.
    This specialization creates additional CSDS objects, one
    for each word that is _not_ annotated in the original corpus file,
    and gives it the default label 'O' (for 'other').
    """

    def __init__(self, corpus_name, corpus_directory):
        super().__init__(corpus_name, corpus_directory)

    def add_o_annotations(self):
        """
        On a second pass through the sentences of the current document,
        creates a CSDS collection with the 'O' label for each word of
        each sentence that has not already been labeled based on an actual
        annotation.
        :return: None.
        """
        tokenizer = SpaceTokenizer()
        for sentence_id, sentence in enumerate(self.sentences):
            o_offset_pairs = list(tokenizer.span_tokenize(sentence))
            includes = []
            for pair in o_offset_pairs:
                include = True
                for annotated_pair in self.sentence_to_annotation_offsets[sentence_id]:
                    if pair[0] >= annotated_pair[0] and pair[1] <= annotated_pair[1]:
                        include = False
                        break
                if include:
                    includes.append(pair)
            for (start, end) in includes:
                cog_state = CSDS(
                    sentence,
                    start,
                    end,
                    "O",
                    sentence[start:end],
                    self.doc_id,
                    sentence_id
                )
                self.csds_collection.add_o_instance(cog_state)

    def add_file(self, xml_file):
        """
        Adds the current file to the growing collection of CSDS objects.
        Includes a second pass to add the 'O' labels.
        :param xml_file: String, name of the current XML file.
        :return: None.
        """
        tree = et.parse(xml_file)
        self.update_nodes_dictionaries(tree)
        self.add_file_to_csds_collection(tree, xml_file)
        self.add_o_annotations()
        self.nodes_to_sentences.clear()
        self.nodes_to_targets.clear()
        self.nodes_to_offsets.clear()
        self.sentences.clear()


# Simple test code.
if __name__ == '__main__':
    # Set verbose to True below to show the CSDS labeled_instances in the output.
    verbose = True
    input_processor = XMLCorpusToCSDSCollection(
        '2010 Language Understanding',
        '../CMU')
    collection = input_processor.create_and_get_collection()
    if verbose:
        for entry in collection.get_next_instance():
            print(entry.get_info_short())
        for entry in collection.get_next_labeled_instance():
            print(entry.get_info_short())
