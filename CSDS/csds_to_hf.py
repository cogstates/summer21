# Owen's experiment to convert a CSDS to the HF data structure

import datasets

from datasets import Dataset

# create a CSDS as dict

csds_dict = {'text' : ["John said he likes beets.", "Mary sometimes says she likes beets.", "Mary maybe likes beets."],
             'class' : ["CB", "NCB", "NCB"]}

csds_dataset = Dataset.from_dict(csds_dict)

print("Done.")


