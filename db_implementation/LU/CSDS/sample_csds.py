# This file creates a sample instance of a CSDS to run tests
# and serves as a unit test of the CSDS API.

from csds import CSDS, CSDSCollection

sample_corpus = [
    ("John said he likes beets.", 5, 9, "CB", "said"),
    ("John said he likes beets.", 13, 18, "NCB", "likes"),
    ("Mary sometimes says she likes beets.", 15, 19, "CB", "says"),
    ("Mary sometimes says she likes beets.", 24, 29, "NCB", "likes"),
    ("Maybe Mulligan said she likes beets.", 15, 19, "NCB", "said"),
    ("Maybe Mulligan said she likes beets.", 24, 29, "NCB", "likes"),
]

sentence_id = 0


def make_cognitive_state(sample_tuple):
    global sentence_id
    csds = CSDS(*sample_tuple, 0, sentence_id)
    sentence_id += 1
    return csds


if __name__ == "__main__":
    sample_csds_collection = CSDSCollection("No text corpus")
    for sentence_id, sample in enumerate(sample_corpus):
        sample_csds_collection.add_labeled_instance(CSDS(*sample, 0, sentence_id))
    print("Created sample CSDS collection")
    print(sample_csds_collection.get_info_short())
    print(sample_csds_collection.get_info_long())
    # Not something you would normally do--therefore not in the API:
    sample_csds_collection.labeled_instances.clear()
    new_samples = list(map(make_cognitive_state, sample_corpus))
    sample_csds_collection.add_list_of_labeled_instances(new_samples)
    print(sample_csds_collection.get_info_short())
    for sample in sample_csds_collection.get_next_instance():
        print(sample.get_marked_text())
