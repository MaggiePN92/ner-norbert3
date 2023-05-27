from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
import torch.nn.functional as F
from collections import Counter
import gzip
import tqdm
from sklearn.model_selection import train_test_split
from conllu import parse
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import gensim
import numpy as np
import pandas as pd
from ner_eval import Evaluator

import warnings
import zipfile
import math
import time
import json
import random

warnings.filterwarnings("ignore")

entities = ["PER", "ORG", "LOC", "GPE_LOC", "GPE_ORG", "PROD", "EVT", "DRV"]


def load_embedding(modelfile):
    # Binary word2vec format:
    if modelfile.endswith(".bin.gz") or modelfile.endswith(".bin"):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=True, unicode_errors="replace"
        )
    # Text word2vec format:
    elif (
            modelfile.endswith(".txt.gz")
            or modelfile.endswith(".txt")
            or modelfile.endswith(".vec.gz")
            or modelfile.endswith(".vec")
    ):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=False, unicode_errors="replace"
        )
    # ZIP archive from the NLPL vector repository:
    elif modelfile.endswith(".zip"):
        with zipfile.ZipFile(modelfile, "r") as archive:
            # Loading and showing the metadata of the model:
            metafile = archive.open("meta.json")
            metadata = json.loads(metafile.read())
            for key in metadata:
                print(key, metadata[key])
            print("============")
            # Loading the model itself:
            stream = archive.open(
                "model.bin"  # or model.txt, if you want to look at the model
            )
            emb_model = gensim.models.KeyedVectors.load_word2vec_format(
                stream, binary=True, unicode_errors="replace"
            )
    else:  # Native Gensim format?
        emb_model = gensim.models.KeyedVectors.load(modelfile)
        #  If you intend to train the model further:
        # emb_model = gensim.models.Word2Vec.load(embeddings_file)
    # Unit-normalizing the vectors (if they aren't already):
    emb_model.init_sims(
        replace=True
    )
    return emb_model


class ConlluDataset(Dataset):
    def __init__(self, sentences, label_vocab=None, embedding_model=None):

        self.embeddings = embedding_model
        self._unk = self.embeddings.key_to_index['<unk>']

        self.text = []

        for sentence in sentences:
            self.text.append([token['form'] for token in sentence])

        self.label = []
        for sentence in sentences:
            self.label.append([token['misc']['name'] for token in sentence])

        self.flat = [item for sublist in self.label for item in sublist]

        if label_vocab == None:
            self.label_vocab = list(set([item for sublist in self.label for item in sublist]))
        else:
            self.label_vocab = label_vocab

        self.label_indexer = {i: n for n, i in enumerate(self.label_vocab)}
        self.inverse_indexer = {n: i for n, i in enumerate(self.label_vocab)}

    def __getitem__(self, index):
        current_text = self.text[index]
        current_label = self.label[index]

        indices = []
        for i in current_text:
            if i in self.embeddings:
                indices.append(self.embeddings.key_to_index[i])
            else:
                indices.append(self._unk)

        X = torch.LongTensor(indices)
        y = torch.LongTensor([self.label_indexer[i] for i in current_label])
        return X, y

    def __len__(self):
        return len(self.text)


def pad_batches(batch, pad_idx):
    longest_sentence = max([X.size(0) for X, y in batch])

    new_X = torch.stack([F.pad(X, (0, longest_sentence - X.size(0)), value=pad_idx) for X, y in batch])
    new_y = torch.stack([F.pad(y, (0, longest_sentence - y.size(0)), value=-1) for X, y in batch])
    return new_X, new_y


def create_class_weight(labels_dict, mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu * (total / float(labels_dict[key])))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight


def f1(precision, recall):
    score = 2 * (precision * recall) / (precision + recall + 1e-6)
    return score


class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_labels, embedding_model):
        super().__init__()

        self.embedder = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings_model.vectors))
        self._hidden = nn.Linear(embedding_model.vector_size, hidden_dim)
        self._output = nn.Linear(hidden_dim, num_labels)

    def forward(self, X):
        # embedding = self.embedder(X, torch.LongTensor([0]))
        embedding = self.embedder(X)
        # print(f"Embedding shape: {embedding.shape}")
        # embedding = embedding.permute(2, 0, 1)
        hidden = self._hidden(embedding)
        hidden = F.relu(hidden)
        # print(f"Hidden shape: {hidden.shape}")
        output = self._output(hidden)
        # output shape: batch_size, text_length, num_labels
        # print(f"Output shape: {output.shape}")

        return output


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--lr", help="learning rate", default=3e-3, type=float)
    parser.add_argument('--epoch', help="number of epoch", default=100, type=int)
    args = parser.parse_args()
    print("lr: {}".format(args.lr))

    # setting a seed for reproducibility
    torch.manual_seed(42)

    # print("Loading the embeddings...\n")

    embeddings_model = load_embedding(
        '/fp/homes01/u01/ec-dongzhuz/IN9550_Assignment_Dongzhuoran_Zhou/assignment3/100/model.bin')

    # embeddings_model.add('<unk>', weights=torch.rand(embeddings_model.vector_size))
    # embeddings_model.add('<pad>', weights=torch.zeros(embeddings_model.vector_size))
    embeddings_model.add_vectors(['<pad>'], np.zeros((1, embeddings_model.vector_size)))
    embeddings_model.add_vectors(['<unk>'], np.random.random((1, embeddings_model.vector_size)))

    pad_idx = embeddings_model.key_to_index['<pad>']

    data_dir = r"/fp/projects01/ec30/IN5550/obligatories/3/norne-nb-in5550-train.conllu.gz"
    f = gzip.open(data_dir, 'rt')
    f_content = f.read()
    data = parse(f_content)
    train_df, val_df = train_test_split(data, test_size=0.1)

    train_dataset = ConlluDataset(train_df, embedding_model=embeddings_model)
    val_dataset = ConlluDataset(val_df, embedding_model=embeddings_model, label_vocab=train_dataset.label_vocab)

    print(f'Number of labels: {len(train_dataset.label_vocab)}')
    # print(train_dataset.inverse_indexer)

    print(f"Number of training exmaples: {len(train_dataset)}")
    print(f"Number of validation exmaples: {len(val_dataset)}")

    ##class weight calculations
    count_label = dict(Counter(train_dataset.flat))
    w = create_class_weight(count_label)
    w['O'] = 0.1
    c = train_dataset.label_indexer
    wl = [w[k] for k in c]
    weights = torch.FloatTensor(wl)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              collate_fn=lambda x: pad_batches(x, pad_idx=pad_idx))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            collate_fn=lambda x: pad_batches(x, pad_idx=pad_idx))

    model = Classifier(embeddings_model.vector_size, num_labels=len(train_dataset.label_vocab),
                       embedding_model=embeddings_model)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    patience_num = 10
    patience_counter = 0
    early_stop = False
    best_val_f1, best_results, best_results_agg = 0, None, None

    train_loss_values, validation_loss_values = [], []

    start_time = time.time()

    print("\n---------------------------------------------------------------\nStart training...\n\n")

    for epoch in range(100):
        model.train()
        for X, y in tqdm.tqdm(train_loader):
            assert X.shape == y.shape, 'shape not matching'
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            y_pred = y_pred.permute(0, 2, 1)
            # y = y.permute(0, 2, 1)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        predictions, true_labels = [], []
        for X, y in tqdm.tqdm(val_loader):
            assert X.shape == y.shape, 'shape not matching'
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            y_pred = y_pred.permute(0, 2, 1)
            # y = y.permute(0, 2, 1)
            val_loss = criterion(y_pred, y)

            x = y_pred.argmax(dim=1).detach().numpy()
            y = y.numpy()
            # print(f"prediction shape: {x.shape}, true shape: {y.shape}")
            predictions.extend(list(p) for p in x)
            true_labels.extend(list(p) for p in y)

        pred_tags = []
        for p, l in zip(predictions, true_labels):
            pred_tags.append([train_dataset.inverse_indexer[p_i] for p_i, l_i in zip(p, l) if l_i != -1])

        valid_tags = []
        for l in true_labels:
            valid_tags.append([train_dataset.inverse_indexer[l_i] for l_i in l if l_i != -1])

        train_loss_values.append(loss.item())
        validation_loss_values.append(val_loss.item())

        evaluator = Evaluator(valid_tags, pred_tags, entities)
        results, results_agg = evaluator.evaluate()
        val_f1 = f1(results["strict"]["precision"], results["strict"]["recall"])

        if epoch % 5 == 0:
            print(f"\tepoch: {epoch + 1}; train loss: {loss.item()}; val loss = {val_loss.item()}")
            print("F1 scores:")
            for entity in results_agg:
                prec = results_agg[entity]["strict"]["precision"]
                rec = results_agg[entity]["strict"]["recall"]
                print(f"{entity}:\t{f1(prec, rec):.4f}")
            print(f"Overall score: {val_f1:.4f}")

        improved_f1 = val_f1 - best_val_f1

        if improved_f1 > 1e-5 or best_val_f1 == 0:
            best_val_f1, best_results, best_results_agg = val_f1, results, results_agg

            if improved_f1 < 0.0002:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        if (patience_counter >= patience_num and epoch > 5):
            print(f'Early stop at epoch: {epoch + 1}')
            break

    end_time = time.time()

    print(f"\n\nIt took {(end_time - start_time) / 60:.3f} minutes.")
    print("Best validation results:")
    # print(classification_report(valid_tags, pred_tags))
    print("F1 scores:")
    for entity in best_results_agg:
        prec = best_results_agg[entity]["strict"]["precision"]
        rec = best_results_agg[entity]["strict"]["recall"]
        print(f"{entity}:\t{f1(prec, rec):.4f}")
    print(f"Overall score: {best_val_f1:.4f}")
    print("\n---------------------------------------------------------------\n")