import shutil
import tempfile
import unittest

from pkg_resources import resource_filename

from jiant.preprocess import build_tasks
from jiant.utils.config import params_from_file


class TestBuildFactualityTask(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.HOCON1 = """
            pretrain_tasks = "meantime,uw"
            target_tasks = "factbank,uds-ih2"
            tokenizer = bert-large-cased
            input_module = bert-large-cased
            reload_tasks = 1
            exp_dir = test_build_fact_task
        """
        self.DEFAULTS_PATH = resource_filename(
            "jiant", "config/defaults.conf"
        )  # To get other required values.
        self.params1 = params_from_file(self.DEFAULTS_PATH, self.HOCON1)

    def test_build_tasks(self):
        pretrain_tasks, target_tasks, vocab, word_embs = build_tasks(self.params1)
        assert pretrain_tasks[0].name == "meantime"
        assert pretrain_tasks[1].name == "uw"
        assert target_tasks[0].name == "factbank"
        assert target_tasks[1].name == "uds-ih2"
        self.assertCountEqual(
            first=pretrain_tasks[0].example_counts, second={"train": 293, "val": 65, "test": 61},
            msg=f"Unexpected number of sentences for {pretrain_tasks[0].name}: " + str(
                pretrain_tasks[0].example_counts))
        self.assertCountEqual(
            first=pretrain_tasks[1].example_counts, second={'train': 2765, 'val': 992, 'test': 265},
            msg=f"Unexpected number of sentences for {pretrain_tasks[1].name}: " + str(
                pretrain_tasks[1].example_counts))
        self.assertCountEqual(
            first=target_tasks[0].example_counts, second={"train": 1899, "val": 709, "test": 197},
            msg=f"Unexpected number of sentences for {target_tasks[0].name}: " + str(target_tasks[0].example_counts))
        self.assertCountEqual(
            first=target_tasks[1].example_counts, second={'train': 9098, 'val': 1246, 'test': 1219},
            msg=f"Unexpected number of sentences for {target_tasks[1].name}: " + str(target_tasks[1].example_counts))

    def tearDown(self) -> None:
        shutil.rmtree("test_build_fact_task")