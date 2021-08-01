# This file creates a sample instance of a CSDS to run tests
# and serves as a unit test of the CSDS API.

from csds import CSDS, CSDSCollection

sample_corpus = [
    ("John said he likes beets.", 5, 9, "CB"),
    ("John said he likes beets.", 13, 18, "NCB"),
    ("Mary sometimes says she likes beets.", 15, 19, "CB"),
    ("Mary sometimes says she likes beets.", 24, 29, "NCB"),
    ("Maybe Mulligan said she likes beets.", 15, 19, "NCB"),
    ("Maybe Mulligan said she likes beets.", 24, 29, "NCB"),
]


def make_cognitive_state(sample_tuple):
    return CSDS(*sample_tuple)


if __name__ == "__main__":
    sample_csds = CSDSCollection("No text corpus")
    for sample in sample_corpus:
        sample_csds.add_instance(CSDS(*sample))
    # print("Created sample CSDS instance")
    # print(sample_csds.get_info_short())
    # print(sample_csds.get_info_long())
    # # Not something you would normally do--therefore not in the API:
    # sample_csds.instances.clear()
    new_samples = list(map(make_cognitive_state, sample_corpus))
    sample_csds.add_list_of_instances(new_samples)
    print(sample_csds.get_info_short())
    for sample in sample_csds.get_next_instance():
        print(sample.get_marked_text())
