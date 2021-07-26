# testing out ElementTree parsing

import xml.etree.ElementTree as et
from CSDS.csds import CognitiveStateFromText, CSDS


def add_file(xml_file, csds_collection):
    tree = et.parse(xml_file)
    text_with_nodes = tree.find('TextWithNodes')
    sentence = text_with_nodes.tail
    nodes_in_sentence = []
    nodes_to_sentences = {}
    nodes_to_targets = {0: text_with_nodes.text}
    for node in text_with_nodes.findall('Node'):
        text = node.tail
        node_id = node.attrib['id']
        nodes_to_targets[node_id] = text
        nodes_in_sentence.append(node_id)
        if '\n' in text:
            parts = text.split('\n')
            sentence += parts[0]
            for node_in_sentence in nodes_in_sentence:
                nodes_to_sentences[node_in_sentence] = sentence
            sentence = parts[-1]
        else:
            sentence += text
    annotation_sets = tree.findall('AnnotationSet')
    for annotation_set in annotation_sets:
        for annotation in annotation_set:
            cog_state = CognitiveStateFromText(nodes_to_sentences[annotation.attrib['StartNode']],
                                               annotation.attrib['StartNode'],
                                               annotation.attrib['EndNode'],
                                               annotation.attrib['Type']
                                               )
            csds_collection.add_instance(cog_state)
            print(cog_state.get_info_short())


if __name__ == '__main__':
    in_file = './20000415_apw_eng-New.xml'
    collection = CSDS('2010 Language Understanding')
    add_file(in_file, collection)
