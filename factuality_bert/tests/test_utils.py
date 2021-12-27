import csv
import os
import shutil
import tempfile
import unittest

import jiant.utils.data_loaders as data_loaders


class TestLoadTsvLabelsOneSentence(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "temp_dataset.tsv")
        with open(self.path, "w") as tsvfile:
            writer = csv.writer(tsvfile, delimiter="\t")
            writer.writerow(["sentence", "label"])
            writer.writerow(["it 's a charming and often affecting journey", 1])
            writer.writerow(["unflinchingly bleak and desperate", 0])

    def test(self):
        max_seq_len = 30
        sent1s, sent2s, labels = data_loaders.load_tsv(
            "MosesTokenizer",
            self.path,
            max_seq_len,
            s1_idx=0,
            s2_idx=None,
            label_idx=1,
            skip_rows=1,
        )
        print(sent2s)
        assert sent2s == []
        assert (
            len(sent1s) == 2
        ), "The length of the set of first sentences != total rows in data file"
        assert len(sent2s) == 0, "Second sentence does not exist yet len(sent2s) != 0"
        assert len(labels) == 2, "The length of labels should be equal to rows in data file"

    def tearDown(self):
        shutil.rmtree(self.temp_dir)


class TestLoadTsvLabelsTwoSentencesReturnIndices(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "temp_dataset_two_sentences.tsv")
        with open(self.path, "w") as tsvfile:
            writer = csv.writer(tsvfile, delimiter="\t")
            writer.writerow(["sentence", "label"])
            writer.writerow(
                ["it 's a charming and often affecting journey", "I agree what's better?", 1]
            )
            writer.writerow(["unflinchingly bleak and desperate", "that's amazing", 0])

    def test(self):
        max_seq_len = 30
        sent1s, sent2s, labels, indices = data_loaders.load_tsv(
            "MosesTokenizer",
            self.path,
            max_seq_len,
            s1_idx=0,
            s2_idx=1,
            return_indices=1,
            label_idx=1,
            skip_rows=1,
        )
        assert "charming" in sent1s[0], "sent1s is not tokenized first sentence"
        assert "agree" in sent2s[0], "sent2s is not tokenized second sentence"
        assert (
            len(sent1s) == 2
        ), "The length of the set of first sentences != total rows in data file"
        assert (
            len(sent2s) == 2
        ), "The length of the set of second sentences != total rows in data file"
        assert len(labels) == 2, "The length of labels should be equal to num rows in data file"
        assert (
            len(indices) == 2
        ), "The length of returned indices should be equal to num rows in data file"

    def tearDown(self):
        shutil.rmtree(self.temp_dir)


class TestLoadDiagnosticDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "temp_dataset_diagnostic.tsv")
        with open(self.path, "w") as tsvfile:
            writer = csv.writer(tsvfile, delimiter="\t")
            writer.writerow(
                [
                    "Lexical Semantics",
                    "Predicate-Argument Structure",
                    "Logic",
                    "Knowledge",
                    "Domain",
                    "Premise",
                    "Hypothesis",
                    "Label",
                ]
            )
            writer.writerow(
                [
                    "",
                    "",
                    "Negation",
                    "",
                    "Artificial",
                    "The cat sat on the mat",
                    "The cat did not sit on the mat.",
                    "contradiction",
                ]
            )

    def test(self):
        max_seq_len = 30
        label_map = {"contradiction": 0, "entailment": 1}

        def label_fn(x):
            return label_map[x]

        output_dictionary = data_loaders.load_diagnostic_tsv(
            "MosesTokenizer",
            self.path,
            max_seq_len,
            s1_col="Premise",
            s2_col="Hypothesis",
            label_col="Label",
            label_fn=label_fn,
        )
        assert (
            len(output_dictionary["sents1"]) == 1
        ), "The length of the set of first sentences != total rows in data file"
        assert (
            len(output_dictionary["sents2"]) == 1
        ), "Second sentence does not exist yet len(sent2s) != 0"
        assert (
            len(output_dictionary["targs"]) == 1
        ), "The length of labels should be equal to rows in data file"
        assert (
            len(output_dictionary["idxs"]) == 1
        ), "The length of labels should be equal to rows in data file"
        assert (
            len(output_dictionary["knowledge"]) == 1
        ), "The length of labels should be equal to rows in data file"
        assert (
            len(output_dictionary["lex_sem"][0]) == 0
        ), "If the field in row is blank in diagnostic dataset, we should return []"
        assert output_dictionary["logic"][0] == [
            0
        ], "If the field in row is not blank, we should return [0], where 0 is index"
        assert "cat" in output_dictionary["sents1"][0], "sent1s output is wrong"
        assert "not" in output_dictionary["sents2"][0]

    def tearDown(self):
        shutil.rmtree(self.temp_dir)


class TestLoadFactualityDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "temp_factuality.conll")
        with open(self.path, "w") as f:
            f.write(
                "\n".join([
                    "0	From	_	_	from	ADP	IN	_	2	case	_	_",
                    "1	the	_	_	the	DET	DT	Definite=Def|PronType=Art	2	det	_	_",
                    "2	AP	_	_	AP	PROPN	NNP	Number=Sing	3	nmod	_	_",
                    "3	comes	3.0	3.0	come	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	3	root	_	_",
                    "4	this	_	_	this	DET	DT	Number=Sing|PronType=Dem	5	det	_	_",
                    "5	story	_	_	story	NOUN	NN	Number=Sing	3	nsubj	_	_",
                    "6	:	_	_	:	PUNCT	:	_	3	punct	_	_",
                ]) + "\n\n" +
                "\n".join([
                    "0	That	_	_	that	PRON	DT	Number=Sing|PronType=Dem	2	nsubj	_	_",
                    "1	alone	_	_	alone	ADJ	JJ	Degree=Pos	0	amod	_	_",
                    "2	makes	2.6	2.6	make	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	2	root	_	_",
                    "3	them	_	_	they	PRON	PRP	Case=Acc|Number=Plur|Person=3|PronType=Prs	2	dobj	_	_",
                    "4	unique	-0.1	-0.1	unique	ADJ	JJ	Degree=Pos	2	xcomp	_	SpaceAfter=No",
                    "5	.	_	_	.	PUNCT	.	_	2	punct	_	_",
                ])
            )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test(self):
        tokenized_records = data_loaders.load_factuality_conll("MosesTokenizer", self.path)
        assert len(tokenized_records) == 2
        expected_records = [
            {'text': 'From the AP comes this story :',
             'targets': [{'span1': [3, 4], 'label': 3.0, 'span_text': 'comes'}],
             'idx': 0},
            {'text': 'That alone makes them unique .',
             'targets': [{'span1': [2, 3], 'label': 2.6, 'span_text': 'makes'},
                         {'span1': [4, 5], 'label': -0.1, 'span_text': 'unique'}],
             'idx': 1}]
        self.assertCountEqual(first=tokenized_records, second=expected_records)
