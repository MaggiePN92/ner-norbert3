from argparse import ArgumentParser
import pandas as pd
import pickle
import zipfile
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils import data
import conllu
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
# from norbert_lstm_crf import ConlluDataset, NorBertCRFModel, collate_fn, build_mask
import transformers
from utils.labels2int import Label2Int
from data.get_encodings import get_encodings_and_labels
from transformers import AutoTokenizer
from conllu import parse, parse_tree
import tqdm
from utils.labels2int import Label2Int
from data.dataset import NERDataset
from data.data_utils import get_data, datasplit
import gzip
from utils.modeling_norbert import NorbertForTokenClassification
if __name__ == "__main__":
    # Add command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--test",
        "-t",
        required=False,
        help="path to a file with test data",
        default="norne-nb-in5550-train.conllu.gz",
    )

    args = parser.parse_args()

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # Get saved model dictionary from saga
    # saved_model = torch.load(r"nortbert3_small_10layers_089f1")
    saved_model = torch.load(
        "/fp/projects01/ec30/magnuspn/oblig3_models/nortbert3_small_10layers_089f1",
        map_location=device
    )

    # Load model parameters
    args.model_name = r"ltg/norbert3-small"

    num_test_data = -1
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name
    )
    test_tokens, test_labels_raw = get_data(
        args.test
    )
    test_tokens, test_labels = test_tokens[:num_test_data], test_labels_raw[:num_test_data]
    test_encodings, test_alligned_labels = get_encodings_and_labels(test_tokens, test_labels, tokenizer)

    test_dataset = NERDataset(test_encodings, test_alligned_labels)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = NorbertForTokenClassification.from_pretrained(
            args.model_name,
            num_labels=17, # len(test_dataset.get_unique_labels()) - 1
            num_hidden_layers=10,
            hidden_dropout_prob=0.2
        )
    model.load_state_dict(saved_model)

    model.to(device)
    model.eval()

    # data = parse(open(args.test, "r").read())
    # test_dataset = ConlluDataset(data, label_vocab=vocab)
    # test_iter = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    predictions = []

    # for text, label in tqdm.tqdm(test_loader):
    #     X = tokenizer(text, is_split_into_words=True, return_tensors='pt', padding=True)['input_ids']
    #     batch_mask = build_mask(tokenizer, X)
    #     X, batch_mask = X.to(device), batch_mask.to(device)
    #     logits = model(X, batch_mask)[0]
    #     label = label.to(device)
    #     y_pred = model.crf.decode(logits, mask=label.gt(-1))
    #     for i in y_pred:
    #         self.inverse_indexer = {n: i for n, i in enumerate(self.label_vocab)}
    #         predictions.append([test_dataset.inverse_indexer[int(p)] for p in i])

    y_preds = []
    y_trues = []
    predictions = []
    true_labels = []
    label2int = Label2Int()
    # inverse_indexer = {n: i for n, i in enumerate(label_vocab)}
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        y_pred = outputs[1].argmax(dim=2).tolist()
        y_preds += y_pred
        y_trues += labels.tolist()
        true_labels = labels.tolist()
        for p_list in y_pred:
            predictions.append([label2int.int2label[p] for p,l in zip (p_list, true_labels) if l != -100 ])

    with gzip.open(args.test, "rt", encoding="utf-8") as f:
        data = f.read()

    parsed_data = conllu.parse(data)

    for i, sentence in enumerate(parsed_data[:num_test_data]):
        sl = len(sentence)
        p = predictions[i][0:sl]
        for j, token in enumerate(sentence):
            try:
                token['misc']['name'] = p[j]
            except:
                pass

    # with open('predictions.conllu', 'w') as f:
    #     f.writelines([sentence.serialize() + "\n" for sentence in parsed_data[:num_test_data]])

    with gzip.open('predictions.conllu.gz', 'wt', encoding="utf-8") as f:
        f.writelines([sentence.serialize() + "\n" for sentence in parsed_data[:num_test_data]])


    # read the prediction
    # prediction_tokens, prediction_labels_raw = get_data(
    #     'predictions.conllu.gz'
    # )
    #
    # print("done")