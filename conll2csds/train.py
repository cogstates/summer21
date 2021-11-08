import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoConfig, AutoTokenizer
from bert_custom import BertEFP
from dataloader import FactbankLoader
from evaluate import evaluate
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--maxlen_train', type=int, default=32, help='Maximum number of tokens in the input sequence during training.')
parser.add_argument('--maxlen_val', type=int, default=32, help='Maximum number of tokens in the input sequence during evaluation.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size during training.')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate for Adam.')
parser.add_argument('--num_eps', type=int, default=2, help='Number of training epochs.')
parser.add_argument('--num_threads', type=int, default=1, help='Number of threads for collecting the datasets.')
parser.add_argument('--output_dir', type=str, default='my_model', help='Where to save the trained model, if relevant.')
parser.add_argument('--model',type=str,default=None)
parser.add_argument('--cluster', type=int, default=2, help = "0 for local, 1 for SeaWulf, 2 for AI Institute")

args = parser.parse_args()

def train(model, criterion, optimizer, train_loader, val_loader, args):
    best_acc = 0
    for epoch in trange(args.num_eps, desc="Epoch"):
        model.train()
        for i, (input_ids, attention_mask, labels) in enumerate(tqdm(iterable=train_loader, desc="Training")):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(input=logits.squeeze(-1), target=labels.float())
            loss.backward()
            optimizer.step()
        val_acc, val_loss = evaluate(model=model, loss=criterion, dataloader=val_loader, device=device)
        print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(epoch, val_acc, val_loss))
        if val_acc > best_acc:
            print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))
            best_acc = val_acc
            model.save_pretrained(save_directory=f'models/{args.output_dir}/')
            config.save_pretrained(save_directory=f'models/{args.output_dir}/')
            tokenizer.save_pretrained(save_directory=f'models/{args.output_dir}/')


if __name__ == "__main__":

    if args.model is None:
        args.model = 'bert-base-uncased'
    if args.cluster == 0:
        file_path_train = "/Users/john/PycharmProjects/summer21/conll2csds/factbank_v1/train.conll"
        file_path_dev = "/Users/john/PycharmProjects/summer21/conll2csds/factbank_v1/dev.conll"
        file_path_test = "/Users/john/PycharmProjects/summer21/conll2csds/factbank_v1/test.conll"
    if args.cluster == 1:
        file_path_train = "/gpfs/scratch/jmurzaku/cogstates/conll2csds/factbank_v1/train.conll"
        file_path_dev = "/gpfs/scratch/jmurzaku/cogstates/conll2csds/factbank_v1/dev.conll"
        file_path_test = "/gpfs/scratch/jmurzaku/cogstates/conll2csds/factbank_v1/test.conll"
    if args.cluster == 2:
        file_path_train = "/home/jmurzaku/cogstates/conll2csds/factbank_v1/train.conll"
        file_path_dev = "/home/jmurzaku/cogstates/conll2csds/factbank_v1/dev.conll"
        file_path_test = "/home/jmurzaku/cogstates/conll2csds/factbank_v1/test.conll"

    config = AutoConfig.from_pretrained(args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BertEFP.from_pretrained('bert-base-uncased')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.HuberLoss()

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    train_set = FactbankLoader(file_path= file_path_train, max_len = args.maxlen_train, tokenizer=tokenizer)
    val_set = FactbankLoader(file_path = file_path_test, max_len = args.maxlen_val, tokenizer=tokenizer)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, num_workers=args.num_threads)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, num_workers=args.num_threads)

    train(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader,
          args=args)

