import xml.etree.ElementTree as ET
from CSDS.csds import CognitiveStateFromText, CSDS

in_file = '/Users/erinye/Desktop/20000415_apw_eng-New.xml'
tree = ET.parse(in_file)

text_with_nodes = tree.find('TextWithNodes')
list = [text_with_nodes.text]
for node in text_with_nodes.findall('Node'):
    list.append(node.tail.replace('\n', ''))
print(list)

annotation_sets = tree.findall('AnnotationSet')
print("length:", len(annotation_sets))
for annotation_set in annotation_sets:
    for node in annotation_set:
        print(node.attrib['StartNode'], node.attrib['EndNode'], node.attrib['Type'])

ds = CSDS("No text corpus")
c = 0
new = {}
for annotation in annotation_sets:
    for node in annotation:
        while c < len(list):
            if not len(list[c]) == ((int(node.attrib['StartNode'])) - (int(node.attrib['EndNode']))):
                print('ERROR: offsets do not match length of snippet')
            ds.instances.append(CognitiveStateFromText(list[c], node.attrib['StartNode'], node.attrib['EndNode'], node.attrib['Type']))
            c += 1
print(ds.get_info_long)

# still need to change offsets so that they start with every new line
# still need to add to dictionary?



