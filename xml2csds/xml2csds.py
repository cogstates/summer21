import glob
import xml.etree.ElementTree as et
from CSDS.csds import CSDS, CSDSCollection


class XMLCorpusToCSDSCollection:
    """
    Class to create a Cognitive State Data Structure (CSDS) collection
    corresponding to a corpus consisting of XML files with annotations
    on text targets (heads) following the GATE format.
    """
    corpus_name = ""
    corpus_directory = ""
    csds_collection = None
    nodes_to_sentences = {}
    nodes_to_targets = {}
    nodes_to_offsets = {}

    def __init__(self, corpus_name, corpus_directory):
        self.corpus_name = corpus_name
        self.corpus_directory = corpus_directory
        self.csds_collection = CSDSCollection(self.corpus_name)

    def update_nodes_dictionaries(self, tree):
        text_with_nodes = tree.find('TextWithNodes')
        nodes_in_sentence = []
        sentence = ""
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
                for node_in_sentence in nodes_in_sentence:
                    self.nodes_to_sentences[node_in_sentence] = sentence
                nodes_in_sentence.clear()
                sentence = parts[-1]
                sentence_length_so_far = len(sentence)
            else:
                sentence += text
                sentence_length_so_far += len(text)

    def add_file_to_csds_collection(self, tree, xml_file):
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
                cog_state = CSDS(self.nodes_to_sentences[annotation.attrib['StartNode']],
                                 head_start,
                                 head_end,
                                 annotation.attrib['Type'],
                                 self.nodes_to_targets[annotation.attrib['StartNode']]
                                 )
                self.csds_collection.add_instance(cog_state)

    def add_file(self, xml_file):
        tree = et.parse(xml_file)
        self.update_nodes_dictionaries(tree)
        self.add_file_to_csds_collection(tree, xml_file)
        self.nodes_to_sentences.clear()
        self.nodes_to_targets.clear()
        self.nodes_to_offsets.clear()

    def create_and_get_collection(self):
        for file in glob.glob(self.corpus_directory + '/*.xml'):
            self.add_file(file)
        return self.csds_collection


if __name__ == '__main__':
    # Set verbose to True below to show the CSDS instances in the output.
    verbose = True
    input_processor = XMLCorpusToCSDSCollection(
        '2010 Language Understanding',
        '../CMU')
    collection = input_processor.create_and_get_collection()
    if verbose:
        for entry in collection.get_next_instance():
            print(entry.get_info_short())
