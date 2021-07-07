# testing out ElementTree parsing

import xml.etree.ElementTree as ET
in_file = '/Users/erinye/Desktop/20000415_apw_eng-New.xml'
tree = ET.parse(in_file)
text_with_nodes = tree.find('TextWithNodes')
print(text_with_nodes.text)
for node in text_with_nodes.findall('Node'):
    print(node.attrib['id'], node.tail.replace('\n', ''))


