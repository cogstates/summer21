import glob
import xml.etree.ElementTree as et
from CSDS.csds import CognitiveStateFromText, CSDS


class XMLCorpusToCSDSCollection:
    corpus_name = ""
    corpus_directory = ""
    csds_collection = None
    nodes_to_sentences = {}
    nodes_to_targets = {}

    def __init__(self, corpus_name, corpus_directory):
        self.corpus_name = corpus_name
        self.corpus_directory = corpus_directory
        self.csds_collection = CSDS(self.corpus_name)

    def update_nodes_dictionaries(self, tree):
        text_with_nodes = tree.find('TextWithNodes')
        nodes_in_sentence = []
        sentence = text_with_nodes.text
        self.nodes_to_targets[0] = text_with_nodes.text
        for node in text_with_nodes.findall('Node'):
            text = node.tail
            node_id = node.attrib['id']
            self.nodes_to_targets[node_id] = text
            nodes_in_sentence.append(node_id)
            if '\n' in text:
                parts = text.split('\n')
                sentence += parts[0]
                for node_in_sentence in nodes_in_sentence:
                    self.nodes_to_sentences[node_in_sentence] = sentence
                nodes_in_sentence.clear()
                sentence = parts[-1]
            else:
                sentence += text

    def add_file_to_csds_collection(self, tree):
        annotation_sets = tree.findall('AnnotationSet')
        for annotation_set in annotation_sets:
            for annotation in annotation_set:
                cog_state = CognitiveStateFromText(self.nodes_to_sentences[annotation.attrib['StartNode']],
                                                   annotation.attrib['StartNode'],
                                                   annotation.attrib['EndNode'],
                                                   annotation.attrib['Type'],
                                                   self.nodes_to_targets[annotation.attrib['StartNode']]
                                                   )
                self.csds_collection.add_instance(cog_state)
                # print(cog_state.get_info_short())

    def add_file(self, xml_file):
        tree = et.parse(xml_file)
        self.update_nodes_dictionaries(tree)
        self.add_file_to_csds_collection(tree)
        self.nodes_to_sentences.clear()
        self.nodes_to_targets.clear()

    def create_and_get_collection(self):
        for file in glob.glob(self.corpus_directory + '/*.xml'):
            self.add_file(file)
        self.add_file('./20000415_apw_eng-New.xml')
        return self.csds_collection


if __name__ == '__main__':
    input_processor = XMLCorpusToCSDSCollection('2010 Language Understanding', '.')
    collection = input_processor.create_and_get_collection()
    for entry in collection.get_next_instance():
        print(entry.get_info_short())
