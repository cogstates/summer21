# testing out ElementTree parsing

import xml.etree.ElementTree as et
from CSDS.csds import CognitiveStateFromText, CSDS

in_file = './20000415_apw_eng-New.xml'
tree = et.parse(in_file)
text_with_nodes = tree.find('TextWithNodes')

nodes = {0: text_with_nodes.text}

for node in text_with_nodes.findall('Node'):
    nodes[node.attrib['id']] = node.tail.replace('\n', '')

annotation_sets = tree.findall('AnnotationSet')

csds_collection = CSDS('2010 Language Understanding')

for annotation_set in annotation_sets:
    for annotation in annotation_set:
        cog_state = CognitiveStateFromText(nodes[annotation.attrib['StartNode']],
                                           annotation.attrib['StartNode'],
                                           annotation.attrib['EndNode'],
                                           annotation.attrib['Type']
                                           )
        csds_collection.add_instance(cog_state)
        print(cog_state.get_info_short())

# test to make sure everything is working as intended
# need to make sure snippets can be added correctly
