from datasets import Dataset, DatasetDict, ClassLabel, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import glob
import xml.etree.ElementTree as et


class CSDS:
    """
    Cognitive States Data Structure (CSDS): Represents information about 
    the writer's cognitive state and the text from which we have gleaned 
    this information.
    """
    text = ""  # sentence in which the annotated head occurs
    head_start = -1  # offset within sentence of start of head word of proposition
    head_end = -1  # offset of end of head word of proposition
    head = ""  # target of annotation within sentence
    belief = ""  # belief value (values are corpus-specific)
    sentiment = ""  # sentiment value (values are corpus-specific)

    def __init__(self, this_text, this_head_start, this_head_end, this_belief, this_head=""):
        self.text = this_text
        self.head_start = this_head_start
        self.head_end = this_head_end
        self.belief = this_belief
        self.head = this_head

    def get_info_short(self):
        return "<CSDS instance (" + str(self.head_start) + ") (" + self.text + ") (" + \
               self.head + ") (" + self.belief + ")> "

    def get_marked_text(self):
        # puts stars around annotated snippet
        new_sentence = self.text[0:self.head_start] + "* " + self.text[self.head_start:self.head_end] + \
                       " *" + self.text[self.head_end: len(self.text)]
        return new_sentence

    def get_belief(self):
        return self.belief


class CSDSCollection:
    instances = []
    corpus = ""

    def __init__(self, this_corpus):
        self.corpus = this_corpus

    # add a single new instance
    def add_instance(self, new_instance):
        self.instances.append(new_instance)

    # add a list of new instances
    def add_list_of_instances(self, list_of_new_instances):
        self.instances.extend(list_of_new_instances)

    # return instances as list
    def get_all_instances(self):
        return self.instances

    # generator: return the next instance in an iteration over the instances
    def get_next_instance(self):
        for instance in self.instances:
            yield instance

    def get_info_short(self):
        return "<CSDS from \"" + self.corpus + "\"; " + str(len(self.instances)) + " instances>"

    def get_info_long(self):
        message = "<CSDS from \"" + self.corpus + "\"; " + str(len(self.instances)) + " instances:\n"
        for instance in self.instances:
            message += "   " + instance.get_info_short() + "\n"
        message += ">\n"
        return message


class XMLCorpusToCSDSCollection:
    """
    Class to create a Cognitive State Data Structure (CSDS) collection
    corresponding to a corpus consisting of XML files with annotations
    on text targets (heads) following the GATE format.
    """
    corpus_name = ""
    corpus_directory = ""
    csds_collection = None
    nodes_to_sentences = {}
    nodes_to_targets = {}
    nodes_to_offsets = {}

    def __init__(self, corpus_name, corpus_directory):
        self.corpus_name = corpus_name
        self.corpus_directory = corpus_directory
        self.csds_collection = CSDSCollection(self.corpus_name)

    def update_nodes_dictionaries(self, tree):
        text_with_nodes = tree.find('TextWithNodes')
        nodes_in_sentence = []
        sentence = ""
        if text_with_nodes.text is not None:
            sentence += text_with_nodes.text
        sentence_length_so_far = len(sentence)
        self.nodes_to_targets[0] = text_with_nodes.text
        for node in text_with_nodes.findall('Node'):
            text = node.tail
            node_id = node.attrib['id']
            if text is None:
                continue
            self.nodes_to_targets[node_id] = text
            nodes_in_sentence.append(node_id)
            self.nodes_to_offsets[node_id] = sentence_length_so_far
            if '\n' in text:
                parts = text.split('\n')
                sentence += parts[0]
                for node_in_sentence in nodes_in_sentence:
                    self.nodes_to_sentences[node_in_sentence] = sentence
                nodes_in_sentence.clear()
                sentence = parts[-1]
                sentence_length_so_far = len(sentence)
            else:
                sentence += text
                sentence_length_so_far += len(text)

    def add_file_to_csds_collection(self, tree, xml_file):
        annotation_sets = tree.findall('AnnotationSet')
        for annotation_set in annotation_sets:
            for annotation in annotation_set:
                if annotation.attrib['Type'] == 'paragraph':
                    continue
                node_id = annotation.attrib['StartNode']
                head_start = self.nodes_to_offsets[node_id]
                target_length = len(self.nodes_to_targets[node_id])
                length_check = int(annotation.attrib['EndNode']) - int(node_id)
                if length_check != target_length:
                    print(f'File: {xml_file} - Node: {node_id} has an end marking mismatch.')
                head_end = head_start + target_length
                cog_state = CSDS(self.nodes_to_sentences[annotation.attrib['StartNode']],
                                 head_start,
                                 head_end,
                                 annotation.attrib['Type'],
                                 self.nodes_to_targets[annotation.attrib['StartNode']]
                                 )
                self.csds_collection.add_instance(cog_state)

    def add_file(self, xml_file):
        tree = et.parse(xml_file)
        self.update_nodes_dictionaries(tree)
        self.add_file_to_csds_collection(tree, xml_file)
        self.nodes_to_sentences.clear()
        self.nodes_to_targets.clear()
        self.nodes_to_offsets.clear()

    def create_and_get_collection(self):
        for file in glob.glob(self.corpus_directory + '/*.xml'):
            self.add_file(file)
        return self.csds_collection



class CSDS2HF:

    training_text = []
    test_text = []
    training_labels = []
    test_labels = []
    unique_labels = []
    csds_collection = None

    def __init__(self, csds_collection):
        self.csds_collection = csds_collection

    def populate_lists(self):
        text = []
        beliefs = []
        for instance in self.csds_collection.get_next_instance():
            text.append(instance.get_marked_text())
            beliefs.append(instance.get_belief())
        self.unique_labels = list(set(beliefs))
        size = len(text)
        size_training = size - size // 4
        self.training_text = text[:size_training]
        self.test_text = text[size_training:]
        self.training_labels = beliefs[:size_training]
        self.test_labels = beliefs[size_training:]

    def get_dataset_dict(self):
        self.populate_lists()
        class_label = ClassLabel(num_classes=len(self.unique_labels), names=self.unique_labels)
        csds_train_dataset = Dataset.from_dict(
            {"text": self.training_text, "labels": list(map(class_label.str2int, self.training_labels))}
        )
        csds_test_dataset = Dataset.from_dict(
            {"text": self.test_text, "labels": list(map(class_label.str2int, self.test_labels))}
        )
        return DatasetDict({'train': csds_train_dataset, 'eval': csds_test_dataset})


def notify(string):
    print(">>>>   ", string, "   <<<<")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == '__main__':
    input_processor = XMLCorpusToCSDSCollection(
        '2010 Language Understanding',
        '../CMU')
    collection = input_processor.create_and_get_collection()
    csds2hf = CSDS2HF(collection)
    csds_datasets = csds2hf.get_dataset_dict()
    notify("Created dataset, now tokenizing dataset")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenized_csds_datasets = csds_datasets.map(tokenize_function, batched=True)
    notify("Done tokenizing dataset")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    metric = load_metric("accuracy")
    notify("Starting training")
    training_args = TrainingArguments("../CSDS/test_trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_csds_datasets['train'],
        eval_dataset=tokenized_csds_datasets['eval'],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    notify("Done training")
    results = trainer.evaluate()
    print(results)
