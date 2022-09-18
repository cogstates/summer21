# author: tyler osborne
# osbornty@bc.edu
# 09/15/2022

import xml.etree.ElementTree as ET

tree = ET.parse('../../raw_corpora/BEST/english/data/annotation/NYT_ENG_20131220.0283.best.xml')
root = tree.getroot()

relation_annotations = root[0][0]
sentiment_annotations = root[0][1]
for relation in relation_annotations:
    print(relation.tag, relation.attrib)
    for child in relation:
        if child.tag == 'trigger':
            print(child.tag, child.attrib, child.text)
        else:
            print(child[0].tag, child[0].attrib)
    print('-------------------------------')
