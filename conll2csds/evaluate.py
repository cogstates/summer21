import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from conll2csds.bert_custom import BertEFP
from conll2csds.dataloader import FactbankLoader
from scipy.stats import pearsonr

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--maxlen_train', type=int, default=128, help='Maximum number of tokens in the input sequence during training.')
parser.add_argument('--maxlen_val', type=int, default=128, help='Maximum number of tokens in the input sequence during evaluation.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size during training.')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate for Adam.')
parser.add_argument('--num_eps', type=int, default=2, help='Number of training epochs.')
parser.add_argument('--num_threads', type=int, default=1, help='Number of threads for collecting the datasets.')
parser.add_argument('--output_dir', type=str, default='my_model', help='Where to save the trained model, if relevant.')
parser.add_argument('--model',type=str,default=None)
parser.add_argument('--cluster', type=int, default=0, help = "0 for local, 1 for SeaWulf, 2 for AI Institute")

args = parser.parse_args()

def correlation(logits, labels):
    r = pearsonr(logits.cpu().detach().numpy().flatten().tolist(), labels.cpu().detach().numpy().tolist())

    return r[0]

def evaluate(model, loss, dataloader, device):
    model.eval()

    mean_r, mean_loss, count = 0, 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            logits = model(input_ids, attention_mask)
            mean_loss += loss(logits.squeeze(-1), labels.float()).item()
            mean_r += correlation(logits, labels)
            count += 1
    print(mean_r)
    return mean_r / count, mean_loss / count


if __name__ == "__main__":

    if args.model is None:
        args.model = 'bert-base-uncased'

    # Configuration for the desired transformer model
    config = AutoConfig.from_pretrained(args.model)

    # Tokenizer for the desired transformer model
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Create the model with the desired transformer model
    model = BertEFP.from_pretrained(args.model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Takes as the input the logits of the positive class and computes the binary cross-entropy
    criterion = nn.HuberLoss()

    val_set = FactbankLoader(file_path="/gpfs/scratch/jmurzaku/cogstates/conll2csds/factbank_v1/dev.conll", max_len=args.maxlen_val, tokenizer=tokenizer)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, num_workers=args.num_threads)

    val_acc, val_loss = evaluate(model=model, loss=criterion, dataloader=val_loader, device=device)
    print("Validation Accuracy : {}, Validation Loss : {}".format(val_acc, val_loss))