
from data.dataset import NERDataset
from data.data_utils import get_data, datasplit
from transformers import AutoTokenizer
from data.get_encodings import get_encodings_and_labels
from torch.utils.data import DataLoader
from transformers import DistilBertForTokenClassification, AdamW
import torch
from argparse import ArgumentParser
from tqdm import tqdm
from eval_func import evaluate_seqs


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-cased"
    )
    tokens, labels = get_data(
        args.path2data
        #"/Users/magnusnytun/Documents/git/in5550/obligatory3/data/norne-nb-in5550-train.conllu.gz"
    )
    train_labels, test_labels, train_tokens, test_tokens = datasplit(
        tokens, labels, test_size=0.75
    )
    train_encodings, train_alligned_labels = get_encodings_and_labels(train_tokens, train_labels, tokenizer)
    train_dataset = NERDataset(train_encodings, train_alligned_labels)

    test_encodings, test_alligned_labels = get_encodings_and_labels(test_tokens, test_labels, tokenizer)
    test_dataset = NERDataset(test_encodings, test_alligned_labels)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = DistilBertForTokenClassification.from_pretrained(
        'ltg/norbert2', num_labels=len(train_dataset.get_unique_labels())
    )
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in tqdm(range(args.epochs)):
        accum_loss = 0
        for batch in tqdm(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            accum_loss += loss.item()
            loss.backward()
            optim.step()
            
        print(f"Epoch {epoch} loss: {accum_loss / len(train_loader)}")

    results, eval_agg = evaluate_seqs(model, test_loader, device)

    print(results)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path2data", type=str, 
                        default="/fp/projects01/ec30/IN5550/obligatories/3/norne-nb-in5550-train.conllu.gz")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    main(args)
