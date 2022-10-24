import spacy
from spacy import displacy

# To either hold on tight or get out, as much of Asia goes into recession.

nlp = spacy.load("en_core_web_sm")
doc = nlp('Oshkosh Truck Corp., Oshkosh, Wis., estimated earnings for its fourth quarter ended Sept. 30 fell 50% to 75% below the year-earlier $4.5 million, or 51 cents a share.')
displacy.serve(doc, style="dep")