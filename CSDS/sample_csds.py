# This file creates a sample instance of a CSDS to run tests.
from csds import CognitiveStateFromText, CSDS

sample_csds = CSDS("No text corpus")

sample_csds.instances.append(CognitiveStateFromText("John said he likes beets.", 5, 9, "CB"))
sample_csds.instances.append(CognitiveStateFromText("John said he likes beets.", 13, 18, "NCB"))
sample_csds.instances.append(CognitiveStateFromText("Mary sometimes says she likes beets.", 15, 19, "CB"))
sample_csds.instances.append(CognitiveStateFromText("Mary sometimes says she likes beets.", 25, 29, "NCB"))
sample_csds.instances.append(CognitiveStateFromText("Maybe Mulligan said she likes beets.", 15, 19, "NCB"))
sample_csds.instances.append(CognitiveStateFromText("Maybe Mulligan said she likes beets.", 25, 29, "NCB"))

print("Created sample CSDS instance")
print(sample_csds.get_info_short())
print(sample_csds.get_info_long())
