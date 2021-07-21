# testing out ElementTree parsing

import xml.etree.ElementTree as ET
from CSDS.csds import CognitiveStateFromText, CSDS

in_file = '/Users/erinye/Desktop/20000415_apw_eng-New.xml'
tree = ET.parse(in_file)
text_with_nodes = tree.find('TextWithNodes')
print(text_with_nodes.text)

for node in text_with_nodes.findall('Node'):
    print(node.attrib['id'], node.tail.replace('\n', ''))

list1 = [text_with_nodes.text]
for node in text_with_nodes.findall('Node'):
    list1.append(node.tail.replace('\n', ''))
print(list1)

annotation_sets = tree.findall('AnnotationSet')
print("length:", len(annotation_sets))
for annotation_set in annotation_sets:
    for node in annotation_set:
        print(node.attrib['StartNode'], node.attrib['EndNode'], node.attrib['Type'])


attempt = CSDS('corpus')

for annotation in annotation_sets:
    for node in annotation:
        cog_state = CognitiveStateFromText('', node.attrib['StartNode'], node.attrib['EndNode'], node.attrib['Type'])
        attempt.add_instance(cog_state)
        print(cog_state.get_info_short())


# test to make sure everything is working as intended
# need to make sure snippets can be added correctly

