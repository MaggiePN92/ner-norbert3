import time
from train_fn import train
from data.dataset import NERDataset
from data.data_utils import get_data, datasplit
from transformers import AutoTokenizer
from data.get_encodings import get_encodings_and_labels
from torch.utils.data import DataLoader
from transformers import AdamW
from utils.modeling_norbert import NorbertForTokenClassification
import torch
from argparse import ArgumentParser
from tqdm import tqdm
from eval_func import evaluate_seqs
import transformers


# from seqeval.metrics import f1_score, precision_score, recall_score


def main(args):
    args.model_name = r"/fp/projects01/ec30/models/norlm/NorBERT2/"
    args.path2data = r"/fp/projects01/ec30/IN5550/obligatories/3/norne-nb-in5550-train.conllu.gz"
    freeze_model = False
    lr = 5e-5
    print("lr", lr, "freeze_model", freeze_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name
    )
    tokens, labels = get_data(
        args.path2data
    )
    train_labels, test_labels, train_tokens, test_tokens = datasplit(
        tokens, labels, test_size=0.2
    )
    train_encodings, train_alligned_labels = get_encodings_and_labels(train_tokens, train_labels, tokenizer)
    train_dataset = NERDataset(train_encodings, train_alligned_labels)

    test_encodings, test_alligned_labels = get_encodings_and_labels(test_tokens, test_labels, tokenizer)
    test_dataset = NERDataset(test_encodings, test_alligned_labels)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # model = NorbertForTokenClassification.from_pretrained(
    #     args.model_name,
    #     num_labels=len(train_dataset.get_unique_labels())
    # )
    # model = transformers.AutoModel.from_pretrained(args.model_name)
    model = transformers.BertForTokenClassification.from_pretrained(args.model_name, num_labels=len(
        train_dataset.get_unique_labels())).to(
        device)

    # freeze_model = True
    # lr = 5e-5
    if freeze_model:
        for param in model.base_model.parameters():
            param.requires_grad = False
        optimizer_grouped_parameters = [
            # {'params': [p for p in model.base_model.parameters()],
            #  'lr': 5e-5},
            {'params': [p for p in model.classifier.parameters()],
             'lr': lr * 5},
        ]
    else:
        for param in model.base_model.parameters():
            param.requires_grad = True
        optimizer_grouped_parameters = [
            {'params': [p for p in model.base_model.parameters()],
             'lr': lr},
            {'params': [p for p in model.classifier.parameters()],
             'lr': lr * 5},
        ]
    optim = AdamW(optimizer_grouped_parameters, lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # optim = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optim, gamma=0.9)
    model.to(device)
    model.train()

    start_time = time.time()
    model = train(model, optim, train_loader, scheduler, test_loader, args.epochs, device)
    print(f" Elapsed training time: {time.time() - start_time}")
    args.save = r"best_model"
    if args.save:
        print(f"Saving the model to {args.save}...")
        model.save_pretrained(args.save)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path2data", type=str, default="/fp/projects01/ec30/IN5550/obligatories/3/norne-nb-in5550-train.conllu.gz")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--model_name", type=str, default="ltg/norbert3-base")
    args = parser.parse_args()
    main(args)
