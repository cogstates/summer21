#from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse

import logging
from CSDS.csds import CSDS, CSDSCollection
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset



train_path = '/Users/john/PycharmProjects/summer21/conll2csds/factbank_v1/train.conll'
dev_path = '/Users/john/PycharmProjects/summer21/conll2csds/factbank_v1/dev.conll'
test_path = '/Users/john/PycharmProjects/summer21/conll2csds/factbank_v1/test.conll'

train_path = "/gpfs/scratch/jmurzaku/cogstates/conll2csds/factbank_v1/train.conll"
dev_path = "/gpfs/scratch/jmurzaku/cogstates/conll2csds/factbank_v1/dev.conll"
test_path = "/gpfs/scratch/jmurzaku/cogstates/conll2csds/factbank_v1/test.conll"


#trai
class ConllCSDS(CSDS):
    head_idx = -1

    def __init__(
            self, this_text, head_idx, this_belief, this_head="",
            this_doc_id=-1, this_sentence_id=-1
    ):
        self.doc_id = this_doc_id
        self.sentence_id = this_sentence_id
        self.text = this_text
        self.head_idx = head_idx
        self.belief = this_belief
        self.head = this_head

    def get_marked_text(self):
        return self.text


def read_conll_data(path):
    csds = CSDSCollection("")

    sentences = []
    b = []
    h = []
    with open(path, 'r') as file:
        sentence_tokens = []
        beliefs = []
        heads = []
        for line in tqdm(file):
            line = line.strip()
            array = line.split('\t')
            if len(array) < 7:
                sentences.append(sentence_tokens)
                b.append(beliefs)
                h.append(heads)
                sentence_tokens = []
                beliefs = []
                heads = []
            else:
                word = (array[1])
                belief = (array[2])
                head = int(array[4])
                sentence_tokens.append(word)
                beliefs.append(belief)
                heads.append(head)
    file.close()
    corpus = []
    labels = {
        '3.0': 'ct',
        '2.0': 'pr',
        '1.0': 'ps',
        '0.0': 'uu',
        '-1.0': 'psm',
        '-2.0': 'prm',
        '-3.0': 'ctm'
    }
    for sent_index, i in enumerate(b):
        for index, ii in enumerate(i):
            if ii != '_':
                sform = sentences[sent_index]
                head_temp = sentences[sent_index][int(h[sent_index][index])]
                replacement = "* " + sentences[sent_index][int(h[sent_index][index])] + " *"
                (sform[int(h[sent_index][index])]) = replacement
                joined = ' '.join(sform)
                b_value = "%s" % labels[ii]
                corpus.append(
                    (joined, int(h[sent_index][index]), b_value, sentences[sent_index][int(h[sent_index][index])]))
                (sform[int(h[sent_index][index])]) = head_temp
    for sentence_id, sample in enumerate(corpus):
        csds.add_labeled_instance(ConllCSDS(*sample, 0, sentence_id))

    text = []
    belief = []
    for instance in csds.get_next_instance():
        text.append(instance.get_marked_text())
        belief.append(instance.get_belief())

    #ataset = Dataset.from_dict({"text": text, "labels": belief})
    return text, belief


train, train_b = read_conll_data(train_path)
print(train_b)
test, test_b = read_conll_data(test_path)
class DataLoaderT5(Dataset):
    def __init__(self, file_path, max_len, tokenizer):
        self.text, self.labels = read_conll_data(file_path)
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self._build()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        for idx in range(len(self.text)):
            input_, target = self.text[idx], self.labels[idx]
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=self.max_len, pad_to_max_length = True, return_tensors="pt"
            )
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=2, pad_to_max_length=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup)

tokenizer = T5Tokenizer.from_pretrained('t5-base')
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl




class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)

    def is_logger(self):
        return True

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids = input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self,
                       epoch=None,
                       batch_idx=None,
                       optimizer=None,
                       optimizer_idx=None,
                       optimizer_closure=None,
                       on_tpu=None,
                       using_native_amp=None,
                       using_lbfgs=None
                       ):
        optimizer.step(optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = DataLoaderT5(train_path, max_len=512, tokenizer=tokenizer)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                                num_workers=4)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = DataLoaderT5(dev_path, max_len=512, tokenizer=tokenizer)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)



logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
  def on_validation_end(self, trainer, pl_module):
    logger.info("***** Validation results *****")
    if pl_module.is_logger():
      metrics = trainer.callback_metrics
      # Log results
      for key in sorted(metrics):
        if key not in ["log", "progress_bar"]:
          logger.info("{} = {}\n".format(key, str(metrics[key])))

  def on_test_end(self, trainer, pl_module):
    logger.info("***** Test results *****")

    if pl_module.is_logger():
      metrics = trainer.callback_metrics

      # Log and save results to file
      output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
      with open(output_test_results_file, "w") as writer:
        for key in sorted(metrics):
          if key not in ["log", "progress_bar"]:
            logger.info("{} = {}\n".format(key, str(metrics[key])))
            writer.write("{} = {}\n".format(key, str(metrics[key])))

args_dict = dict(
    data_dir="", # path for data files
    output_dir="/gpfs/scratch/jmurzaku", # path to save the checkpoints
    model_name_or_path='t5-base',
    tokenizer_name_or_path='t5-base',
    max_seq_length=512,
    learning_rate=3e-5,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    n_gpu=1,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)
args = argparse.Namespace(**args_dict)
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=args.output_dir, monitor="val_loss", mode="min", save_top_k=5
)
train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    precision= 16 if args.fp_16 else 32,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)

data  = DataLoaderT5(tokenizer=tokenizer, file_path=test_path, max_len=512)
data = data[42]
emotions = [ "sadness", "joy", "love", "anger", "fear", "surprise"]
for em in emotions:
  print(len(tokenizer.encode(em, padding=True)))

print(tokenizer.decode(data['source_ids']))
print(tokenizer.decode(data['target_ids']))

model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)
trainer.fit(model)
model.model.save_pretrained('/gpfs/scratch/jmurzaku/cogstates/t5_efp')

test_dataset = DataLoaderT5(tokenizer=tokenizer, file_path=test_path, max_len=512)
loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

from scipy.stats import  pearsonr
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
model.model.eval()
outputs = []
targets = []
for batch in tqdm(loader):
    outs = model.model.generate(input_ids=batch['source_ids'],
                                attention_mask=batch['source_mask'],
                                max_length=2)

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

    outputs.extend(dec)
    targets.extend(target)
print(outputs)
print(targets)

precision, recall, f1, _ = precision_recall_fscore_support(targets, outputs, average=None)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)
print("Accuracy: ", accuracy_score(targets, outputs))