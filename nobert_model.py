from email.policy import default
import torch
import transformers
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
import tqdm
from conllu import parse
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
# from data.get_encodings import get_encodings_and_labels
from collections import Counter
import math
import time
from ner_eval import Evaluator
import gzip
from argparse import ArgumentParser
from data.get_encodings import AlignLabelsWithTokens
import pathlib
from typing import List, Tuple
import json

warnings.filterwarnings("ignore")
entities = ["PER", "ORG", "LOC", "GPE_LOC", "GPE_ORG", "PROD", "EVT", "DRV"]

torch.manual_seed(6789)


class AlignLabelsWithTokens:
    def __init__(self):
        """Aligns the labels with the tokens."""
        self._load_mapping()

    def _load_mapping(self):
        """Loads the mapping from the json file."""
        path2mapping = pathlib.Path("obligatory3/data/str2int.json")
        if not path2mapping.exists():
            path2mapping = pathlib.Path("data/str2int.json")
        if not path2mapping.exists():
            path2mapping = pathlib.Path("str2int.json")
        if not path2mapping.exists():
            raise FileNotFoundError(
                "Could not find str2int.json. Should be in obligatory3/data/ or data/ or in the root directory."
            )
        self.mapping = json.load(open(path2mapping))

    def labels2int(self, str_label) -> int:
        """Converts the string label to an integer."""
        return self.mapping[str_label]

    def __call__(
            self, labels: List[str], word_ids: List[str]
    ) -> List[int]:
        """Aligns the labels with the tokens intra doc.

        Args:
            labels (List[str]): unaligned labels
            word_ids (List[str]): byte encoded tokens

        Returns:
            List[int]: aligned labels
        """
        new_labels = []
        labels = labels.numpy()
        for word_id in word_ids:
            if word_id is None:
                new_labels.append(-100)
            else:
                # label = self.labels2int(labels[word_id])
                label = labels[word_id]
                new_labels.append(label)

        return new_labels
def get_encodings_and_labels(
        tokens: List[str], labels: List[str], tokenizer
) -> Tuple[List[int], List[int]]:
    """Encodes the tokens and aligns the labels with the tokens. Also reads the mapping from str to int,
    located in obligatory3/data/str2int.json.

    Args:
        tokens (List[str]): list of tokens
        labels (List[str]): list of labels
        tokenizer : tokenizer from transformers, must be fast.

    Returns:
        Tuple[List[int], List[int]]: encoded tokens and aligned labels
    """

    align_labels_with_tokens = AlignLabelsWithTokens()

    encodings = tokenizer(
        tokens,
        # truncation=True,
        is_split_into_words=True,
        padding=True,
        # max_length=200,
    )

    alligned_labels = []
    for i, label in enumerate(labels):
        word_ids = encodings.word_ids(i)
        alligned_labels.append(align_labels_with_tokens(label, word_ids))

    return encodings, alligned_labels
def f1(precision, recall):
    score = 2 * (precision * recall) / (precision + recall + 1e-6)
    return score


class ConlluDataset(Dataset):
    def __init__(self, sentences, label_vocab=None):
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

        X = current_text
        y = torch.LongTensor([self.label_indexer[i] for i in current_label])
        return X, y

    def __len__(self):
        return len(self.text)




def collate_fn(batch,tokenizer):
    # longest_y = max([y.size(0) for X, y in batch])
    X_raw = [X for X, y in batch]
    y = [y for X, y in batch]
    # y = torch.stack([F.pad(y, (0, longest_y - y.size(0)), value=-100) for X, y in batch])
    # X_all = tokenizer(X, is_split_into_words=True, return_tensors='pt', padding=True)
    # X = tokenizer(X, is_split_into_words=True, return_tensors='pt', padding=True)['input_ids']
    X, y = get_encodings_and_labels(X_raw, y, tokenizer)
    X = tokenizer(X_raw, is_split_into_words=True, return_tensors='pt', padding=True)['input_ids']
    y = torch.stack([torch.tensor(item)for item in y])
    return X, y


def build_mask(tokenizer, ids):
    tok_sents = [tokenizer.convert_ids_to_tokens(i) for i in ids]
    mask = []
    l = tokenizer.all_special_tokens
    l.remove('[UNK]')
    for sentence in tok_sents:
        current = []
        for n, token in enumerate(sentence):
            if token in l or token.startswith('##'):
                # continue
                current.append(n)
            else:
                current.append(n)
        mask.append(current)

    mask = tokenizer.pad({'input_ids': mask}, return_tensors='pt')['input_ids']
    return mask


def create_class_weight(labels_dict, mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu * (total / float(labels_dict[key])))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight


class NorBertModel(nn.Module):
    def __init__(self, norbert, num_labels):
        super().__init__()
        self._bert = transformers.AutoModel.from_pretrained(norbert)

        self._head = nn.Linear(768, num_labels)

    def forward(self, batch, mask):
        # print("mask shape", mask.shape)
        # print("mask", mask)
        b = self._bert(batch)

        # pooler shape: batch_size, t, bert_dim
        # pooler = b.last_hidden_state[:, mask].diagonal().permute(2, 0, 1)
        pooler = b.last_hidden_state
        # pooler = b.last_hidden_state.diagonal().permute(2, 0, 1)
        # print("pooler shape", pooler.shape)
        logits = self._head(pooler)
        # print("logits shape", logits.shape)
        return logits


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--save", help="save model: yes, or no", default="no")
    parser.add_argument("--fine_tune", help="whether to update NorBERT's parameters or not", default="no")
    parser.add_argument("--lr", help="learning rate", default=3e-5, type=float)
    parser.add_argument('--epoch', help="number of epoch", default=60, type=int)
    parser.add_argument('--local', default='fox')

    args = parser.parse_args()

    fine_tune = (args.fine_tune == 'yes')
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    data_dir = r"norne-nb-in5550-train.conllu.gz"
    f = gzip.open(data_dir, 'rt',encoding="utf8")
    f_content = f.read()
    data = parse(f_content)

    norbert_path = r"NorBERT2/"
    # tokenizer = transformers.BertTokenizer.from_pretrained(norbert)
    tokenizer = transformers.AutoTokenizer.from_pretrained(norbert_path)

    train_df, val_df = train_test_split(data, test_size=0.9)

    train_dataset = ConlluDataset(train_df)
    val_dataset = ConlluDataset(val_df, label_vocab=train_dataset.label_vocab)

    ##class weight calculations
    count_label = dict(Counter(train_dataset.flat))
    w = create_class_weight(count_label)
    # print("w", w)
    w['O'] = 0.1
    c = train_dataset.label_indexer
    wl = [w[k] for k in c]
    weights = torch.FloatTensor(wl)
    weights = weights.to(device)

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    # change here, output token index x and aligned y
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: collate_fn(x,tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: collate_fn(x,tokenizer))
    # if args.local == 'fox':
    #     norbert = "../a3_fox/216/"
    # elif args.local == 'saga':
    #     norbert = "/cluster/shared/nlpl/data/vectors/latest/216/"
    # else:
    #     norbert = "/Users/huiliny/Downloads/216/"
    norbert_path = r"NorBERT2/"
    # tokenizer = transformers.BertTokenizer.from_pretrained(norbert)
    # tokenizer = transformers.AutoTokenizer.from_pretrained(norbert_path)
    model = NorBertModel(norbert=norbert_path, num_labels=len(train_dataset.label_vocab))
    # print("num_labels", len(train_dataset.label_vocab))
    # print("labels", train_dataset.label_vocab)
    # model = transformers.AutoModel.from_pretrained(norbert_path)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=weights)

    if fine_tune:
        bert_optimizer = list(model._bert.named_parameters())
        classifier_optimizer = list(model._head.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': args.lr * 5, 'weight_decay': 0.01},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': args.lr * 5, 'weight_decay': 0.0}
        ]
    else:
        # pass
        classifier_optimizer = list(model._head.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': args.lr * 5, 'weight_decay': 0.01},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': args.lr * 5, 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    # optimizer = AdamW(model.parameters(), lr=3e-5)
    train_steps_per_epoch = len(train_dataset) // 32
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(args.epoch // 10) * train_steps_per_epoch,
                                                num_training_steps=args.epoch * train_steps_per_epoch)

    model.to(device)

    patience_num = 10
    patience_counter = 0
    early_stop = False
    best_val_f1, best_results, best_results_agg = 0, None, None

    train_loss_values, validation_loss_values = [], []

    start_time = time.time()

    print("\n---------------------------------------------------------------\nStart training...\n\n")
    align_labels_with_tokens = AlignLabelsWithTokens()
    for epoch in range(args.epoch):
        model.train()
        for X, y in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            # X_all = tokenizer(X, is_split_into_words=True,return_tensors='pt', padding=True)
            # X = tokenizer(X, is_split_into_words=True, return_tensors='pt', padding=True)['input_ids']
            # alligned_labels = []
            # for i, labels in enumerate(y):
            #     word_ids = X_all.word_ids(i)
            #     alligned_label = align_labels_with_tokens(labels, word_ids)
            #     alligned_labels.append(alligned_label)

            batch_mask = build_mask(tokenizer, X)
            X, batch_mask = X.to(device), batch_mask.to(device)
            # print("X shape", X.shape)
            # print("X", X)
            # X, y = get_encodings_and_labels(X, y, tokenizer)
            y = y.to(device)
            # print("y shape", y.shape)
            # print("y", y)
            # print("batch mask shape", batch_mask.shape)
            # print("batch mask", batch_mask)
            y_pred = model(X, batch_mask)

            y_pred = y_pred.permute(0, 2, 1)
            # print("y_pred shape", y_pred.shape)
            # print("y_pred", y_pred)
            loss = criterion(y_pred, y)
            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5)
            optimizer.step()
            scheduler.step()

        model.eval()
        predictions, true_labels = [], []
        for X, y in tqdm.tqdm(val_loader):
            # X = tokenizer(X, is_split_into_words=True, return_tensors='pt', padding=True)['input_ids']
            batch_mask = build_mask(tokenizer, X)
            X, batch_mask = X.to(device), batch_mask.to(device)
            y = y.to(device)
            y_pred = model(X, batch_mask)
            y_pred = y_pred.permute(0, 2, 1)
            val_loss = criterion(y_pred, y)

            x = y_pred.to('cpu').argmax(dim=1).detach().numpy()
            y = y.to('cpu').numpy()
            predictions.extend(list(p) for p in x)
            true_labels.extend(list(p) for p in y)

        # if best_val_loss is None or best_val_loss < val_loss:
        #    best_val_loss, best_val_epoch = val_loss, epoch
        # if best_val_epoch < epoch - max_stagnation:
        #    early_stop = True
        #    print(f'Early stop at epoch: {epoch+1}')
        #    break

        pred_tags = []
        for p, l in zip(predictions, true_labels):
            pred_tags.append([train_dataset.inverse_indexer[p_i] for p_i, l_i in zip(p, l) if l_i != -100])

        valid_tags = []
        for l in true_labels:
            valid_tags.append([train_dataset.inverse_indexer[l_i] for l_i in l if l_i != -100])

        train_loss_values.append(loss.item())
        validation_loss_values.append(val_loss.item())

        evaluator = Evaluator(valid_tags, pred_tags, entities)
        results, results_agg = evaluator.evaluate()
        val_f1 = f1(results["strict"]["precision"], results["strict"]["recall"])

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
            if args.save != 'no':
                print('Saving best model...')
                torch.save({
                    'model': model.state_dict(),
                    'label_vocab': train_dataset.label_vocab, 'label_indexer': train_dataset.label_indexer,
                    'inverse_indexer': train_dataset.inverse_indexer}, f"best_model.pt")

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
    print("F1 scores:")
    for entity in best_results_agg:
        prec = best_results_agg[entity]["strict"]["precision"]
        rec = best_results_agg[entity]["strict"]["recall"]
        print(f"{entity}:\t{f1(prec, rec):.4f}")
    print(f"Overall score: {best_val_f1:.4f}")
    print("\n---------------------------------------------------------------\n")

#    if args.save != "no":
#        print("Saving model...")
#        torch.save({
#            'model': model.state_dict(),
#            'label_vocab': train_dataset.label_vocab, 'label_indexer': train_dataset.label_indexer, 'inverse_indexer': train_dataset.inverse_indexer}, f"saved_model.pt")

