import xml.etree.ElementTree as ET

in_file = '20000415_apw_eng-New.xml'
tree = ET.parse(in_file);
text_with_nodes = tree.find('TextWithNodes')
print(text_with_nodes.text)
for node in text_with_nodes.findall('Node'):
    print(node.attrib['id'], node.tail.replace('\n', ''))

# Dictionary from value of StartNode to (<value of EndNode>, annotation, text)
# Iterate through the Node elements in TextWithNodes to fill in the text field and
# also check the length against EndNode - StartNode and report an error
# discrepancy.

# Dictionary from value of StartNode to CSDS object.

for annotation in annotation_set:
    head_start = annotation.attrib['NodeStart']
    if not head_start in nodes_to_sentences:
        print('ERROR: missing sentence entry for node ' + head_start)
        continue
    if not head_start in nodes_to_snippets:
        print('ERROR: missing snippet entry for node ' + head_start)
        continue
    csds = CognitiveStateFromText(nodes_to_sentences[head_start],
            nodes_to_head_snippets[head_start], 
            head_start, annotation.attrib['NodeEnd'], annotation.attrib['Type'])
    csdss[head_start] = csds
    # Check whether length of text is head_end - head_start


sentence = twn.text
nodes = []
nodes_to_sentences = {}
nodes_to_head_snippets = {}
for node in twn:
    text = node.tail
    id = node.attrib['id']
    nodes_to_head_snippets[id] = text
    nodes.append(id)
    if text.contains('\n'):
        snippets = text.split('\n')
        sentence.append(snippets[0])
        for node in nodes:
            nodes_to_sentences[node] = sentence
        sentence = snippets[-1]
        nodes = []
    else:
        sentence.append(text)
    

    
