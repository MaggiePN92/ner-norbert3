import time
from train_fn import train
from data.dataset import NERDataset
from data.data_utils import get_data, datasplit
from transformers import AutoTokenizer, AutoConfig, AutoModel
from data.get_encodings import get_encodings_and_labels
from torch.utils.data import DataLoader
from transformers import AdamW
import torch
from torch import nn
from argparse import ArgumentParser
from tqdm import tqdm
from utils.seed_everything import seed_everything

def test_model(model, tokenizer, test_input):
    # encode_plus: Returns a dictionary containing the encoded sequence or sequence pair 
    # and additional information: the mask for sequence classification and the 
    # overflowing elements if a max_length is specified.
    encoded_test_input = tokenizer.encode_plus(test_input)
    true_labels = [-100,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,-100]

    print(f'Input Size: {len(encoded_test_input["input_ids"])}')
    print(f'Attention Mask Size: {len(encoded_test_input["attention_mask"])}')
    print(f'True Labels Size: {len(true_labels)}')
    # Unsqueeze simulate a batch of 1
    out = model(torch.tensor(encoded_test_input["input_ids"]).unsqueeze(0), 
                torch.tensor(encoded_test_input["attention_mask"]).unsqueeze(0),
                labels = torch.tensor(true_labels).unsqueeze(0))
    
    #print(encoded_test_input["input_ids"])
    #print(tokenizer.decode(encoded_test_input["input_ids"]))
    #print(f"Length: {len(encoded_test_input['input_ids'])}")
    #print(out[0].shape)
    #print(out[0][0])
    #print(out[0][0].shape)

class NorbertMLP(nn.Module):
    def __init__(self, config, n_layers=0, dropout=0.1):
        super().__init__()
        self.norbert3 = AutoModel.from_config(config)
        self.loss_fn = nn.CrossEntropyLoss()

        self.input_layer = nn.Linear(config.hidden_size, config.hidden_size)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size)
            for _ in range(n_layers)
        ])

        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(config.hidden_size, config.num_labels)
        )

    def forward(self, x, attention_mask, labels):
        x = self.norbert3(x, attention_mask)[0]

        x = self.input_layer(x).relu()

        for layer in self.hidden_layers:
            x = x + layer(x).relu()

        x = self.output_layer(x)

        print(f'Output Shape: {x.shape}')
        print(f'Labels Shape: {labels.shape}')

        # (Batch_Size, Seq_Length, n_classes), (Batch Size, Seq_Length) -> (Batch_Size, n_classes, Seq_Length), (Batch Size, Seq_Length)
        loss = self.loss_fn(torch.transpose(x, 1, 2), labels) 

        # Train function expects loss first, then x.
        return loss, x

def main(args):

    seed_everything(args.seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize a tokenizer. Default: "ltg/norbert3-base"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        truncation=True,
        padding=True
    )
    tokens, labels = get_data(
        args.localpath2data
    )
    train_labels, test_labels, train_tokens, test_tokens = datasplit(
        tokens, labels, test_size=1-args.train_size
    )

    # Train set: Byte-pair encoded training data with corresponding labels
    train_encodings, train_aligned_labels = get_encodings_and_labels(train_tokens, train_labels, tokenizer)
    train_dataset = NERDataset(train_encodings, train_aligned_labels)
    
    # Test set: Byte-pair encoded training data with corresponding labels
    test_encodings, test_aligned_labels = get_encodings_and_labels(test_tokens, test_labels, tokenizer)
    test_dataset = NERDataset(test_encodings, test_aligned_labels)
    
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = len(train_dataset.get_unique_labels())
    model = NorbertMLP(config, n_layers=args.hidden_layers, dropout=args.dropout)
    
    #test_model(model, tokenizer, "Denne setningen ble skrevet bare for testing.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optim = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
    model.to(device)
    model.train() # Enable Dropout

    start_time = time.time()
    model = train(model, optim, train_loader, scheduler, test_loader, args.epochs, device)
    print(f" Elapsed training time: {time.time() - start_time}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path2data", type=str, default="/fp/projects01/ec30/IN5550/obligatories/3/norne-nb-in5550-train.conllu.gz")
    parser.add_argument( "--localpath2data", type=str, 
                    default="C:/Users/olive/Documents/Programming_Environment/MCT/IN5550/datasets/norne-nb-in5550-train.conllu.gz")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="ltg/norbert3-base")
    parser.add_argument("--learning_rate", type=float, default=5e-5),
    parser.add_argument("--train_size", type=float, default=0.8),
    parser.add_argument("--dropout", type=float, default=0.1),
    parser.add_argument("--hidden_layers", type=int, default=0),
    parser.add_argument("--batch_size", type=int, default=8),
    parser.add_argument("--seed", type=int, default=5550)
    args = parser.parse_args()

    main(args)