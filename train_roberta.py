import torch
from transformers import RobertaForTokenClassification, AutoTokenizer
from argparse import ArgumentParser
from utils.seed_everything import seed_everything
from utils.metrics import save_metrics
from data.data_utils import get_data, datasplit
from data.dataset import NERDataset
from data.get_encodings import get_encodings_and_labels
from torch.utils.data import DataLoader
import time
from transformers import AdamW
from train_fn import train

def main(args):

    seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            add_prefix_space=True
        )
    tokens, labels = get_data(
        args.path2data
    )
    train_labels, test_labels, train_tokens, test_tokens = datasplit(
        tokens, labels, test_size=1-args.train_size
    )
    train_encodings, train_alligned_labels = get_encodings_and_labels(train_tokens, train_labels, tokenizer)
    train_dataset = NERDataset(train_encodings, train_alligned_labels)

    test_encodings, test_alligned_labels = get_encodings_and_labels(test_tokens, test_labels, tokenizer)
    test_dataset = NERDataset(test_encodings, test_alligned_labels)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = RobertaForTokenClassification.from_pretrained(
        args.model_name, 
        num_labels=len(train_dataset.get_unique_labels())
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    optim = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optim, gamma=0.9)
    model.to(device)
    model.train()

    start_time = time.time()
    model = train(model, optim, train_loader, scheduler, test_loader, args.epochs, device)
    training_time = time.time() - start_time
    print(f" Elapsed training time: {training_time}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path2data", type=str, default="/fp/projects01/ec30/IN5550/obligatories/3/norne-nb-in5550-train.conllu.gz")
    parser.add_argument( "--localpath2data", type=str, 
                        default="C:/Users/olive/Documents/Programming_Environment/MCT/IN5550/datasets/norne-nb-in5550-train.conllu.gz")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="roberta-base"),
    parser.add_argument("--learning_rate", type=float, default=5e-5),
    parser.add_argument("--train_size", type=float, default=0.8),
    parser.add_argument("--seed", type=int, default=5550)
    args = parser.parse_args()
    main(args)