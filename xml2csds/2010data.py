import xml.etree.ElementTree as ET
from CSDS.csds import CognitiveStateFromText, CSDS

in_file = '/Users/erinye/Desktop/20000415_apw_eng-New.xml'
tree = ET.parse(in_file)

text_with_nodes = tree.find('TextWithNodes')
print(text_with_nodes.text)
list = [text_with_nodes.text]
for node in text_with_nodes.findall('Node'):
    list.append(node.tail.replace('\n', ''))

annotation_sets = tree.findall('AnnotationSet')
print("length:", len(annotation_sets))
for annotation_set in annotation_sets:
    for node in annotation_set:
        print(node.attrib['StartNode'], node.attrib['EndNode'], node.attrib['Type'])

c = CSDS("No text corpus")
new = {}
for annotation in annotation_sets:
    for node in annotation:
        while c < len(list):
            c.instances.append(CognitiveStateFromText(list[c], node.attrib['StartNode'], node.attrib['EndNode'], node.attrib['Type'], ''))
            c += 1

# still need to check that StartNode - EndNode = length of snippet
# still need to synthesize sentences
# still need to change offsets so that they start with every new line



