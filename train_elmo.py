from simple_elmo import ElmoModel
from data.data_utils import get_data, datasplit
from argparse import ArgumentParser
from data.get_encodings import get_encodings_and_labels
import pathlib
from typing import List, Tuple
import json
from transformers import AutoTokenizer
# from data.dataset import NERDataset
import torch
from utils.modeling_norbert import NorbertForTokenClassification,NorbertLSTMForTokenClassification, NorbertELMOForTokenClassification
from transformers import AdamW
from transformers import AutoTokenizer, AutoConfig, AutoModel
from torch.utils.data import DataLoader
import time

# from train_fn import train
# from eval_func import evaluate_seqs
from tqdm import tqdm
from utils.eary_stopping import EarlyStopping
import logging
import torch
from typing import List, Tuple
import numpy as np
# from utils.ner_eval import Evaluator
from utils.labels2int import Label2Int
import torch
from seqeval.metrics import classification_report

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

from seqeval.metrics import classification_report


@torch.no_grad()
def evaluate_seqs(model, dataloader, device, label2int=Label2Int()):
    model.eval()

    y_preds = []
    y_trues = []
    for batch,encodings_elmo  in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        sequence_output = encodings_elmo.to(device)
        # Model Outputs: [0] Loss, [1] Logits, [2] Hidden States, [3] Attention Weights
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels, sequence_output=sequence_output)

        # outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        y_preds += outputs[1].argmax(dim=2).tolist()
        y_trues += labels.tolist()

    # Create list of predicted labels from logit indeces.
    true_preds = [
        [label2int.int2label[p] for (p, l) in zip(y_pred, y_true) if l != -100]
        for y_pred, y_true in zip(y_preds, y_trues)
    ]
    true_labels = [
        [label2int.int2label[l] for (p, l) in zip(y_pred, y_true) if l != -100]
        for y_pred, y_true in zip(y_preds, y_trues)
    ]

    # evalutor = Evaluator(true_labels, true_preds, list(label2int.mapping.keys()))
    # results, eval_agg = evalutor.evaluate()

    # f1_score = f1_score(true_labels, true_preds, mode="strict")
    # precision_score = precision_score(true_labels, true_preds, mode="strict")
    # recall_score = recall_score(true_labels, true_preds, mode="strict")
    # return f1_score, precision_score, recall_score

    clf_rep_str = classification_report(true_labels, true_preds, mode="strict")
    clf_rep_dict = classification_report(true_labels, true_preds, mode="strict", output_dict=True)
    print(clf_rep_str)
    return clf_rep_str, clf_rep_dict


def f1(precision, recall):
    score = 2 * (precision * recall) / (precision + recall + 1e-6)
    return score

def train(
        model, optim, train_loader, scheduler,
        test_loader, n_epochs=5, device="cuda"
):
    accum_loss = []
    model.to(device)
    early_stopper = EarlyStopping()
    prev_macro_f1 = 0.0
    best_clf = 0
    logger.info(f"Starting training with {n_epochs} epochs.")
    for epoch in tqdm(range(n_epochs)):
        model.train()
        accum_loss = 0
        for batch,encodings_elmo in tqdm(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # sequence_output = batch['encodings_elmo'].to(device)
            # sequence_output = sequence_output.astype(np.float32)
            # sequence_output = torch.from_numpy(sequence_output).float().to(device)
            # sequence_output = batch['encodings_elmo'].to(device)
            sequence_output = encodings_elmo.to(device)
            # Model Outputs: [0] Loss, [1] Logits, [2] Hidden States, [3] Attention Weights
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels,sequence_output=sequence_output)
            loss = outputs[0]
            accum_loss += loss.item()
            loss.backward()
            optim.step()

        logger.info(f"Epoch {epoch} loss: {accum_loss / len(train_loader)}")
        clf_str, clf_dict = evaluate_seqs(model, test_loader, device)
        epoch_macro_f1 = clf_dict["macro avg"]["f1-score"]
        logger.info(f"Epoch {epoch} macro f1: {epoch_macro_f1:.3f}")

        # if current model is better then previous best; save state dict

        if epoch_macro_f1 > prev_macro_f1:
            best_state = model.state_dict()
            prev_macro_f1 = epoch_macro_f1
            best_clf = clf_str
        # early stopping returns false if model has not improved for k epochs
        if early_stopper.early_stop(epoch_macro_f1):
            break

        # adjust learning rate
        scheduler.step()

    logger.info("Training finished.")
    logger.info(best_clf)
    # after training; set model to best version in train loop
    # model.load_state_dict(best_state) # TODO
    return model

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

        for word_id in word_ids:
            if word_id is None:
                new_labels.append(-100)
            else:
                label = self.labels2int(labels[word_id])
                new_labels.append(label)

        return new_labels

def get_encodings_and_labels_elmo(
        tokens: List[str], labels: List[str], tokenizer, elmo_model
) :
    # -> Tuple[List[int], List[int],List[float]]
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

    # PATH_TO_ELMO = "norelmo30"
    # elmo_model = ElmoModel()
    # elmo_model.load(PATH_TO_ELMO)


    # encodings_elmo = elmo_model.get_elmo_vectors(tokens)

    encodings = tokenizer(
        tokens,
        truncation=True,
        is_split_into_words=True,
        padding=True,
        max_length=200,
    )

    alligned_labels = []
    for i, label in enumerate(labels):
        word_ids = encodings.word_ids(i)
        alligned_labels.append(align_labels_with_tokens(label, word_ids))

    inputs_text = list()
    for token, input_id in zip(tokens,encodings['input_ids']):
        input_text = tokenizer.convert_ids_to_tokens(input_id,skip_special_tokens=False)
        input_text[1:len(token)+1] = token
        inputs_text.append(input_text)
    encodings_elmo = elmo_model.get_elmo_vectors(inputs_text)
    return encodings, alligned_labels,encodings_elmo


class NERDataset_ForElmo(torch.utils.data.Dataset):
    def __init__(self, encodings, labels,tokens,encodings_elmo) -> None:
        """Dataset class for NER token classification task. Returns aligned input ids and labels.

        Args:
            path2data (str): path to conllu file
            tokenizer : tokenizer from transformers, must be fast.
        """
        super().__init__()
        self.encodings = encodings
        self.labels = labels
        self.tokens = tokens
        self.encodings_elmo = encodings_elmo

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        """Returns the input ids and the labels for the given index."""
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        # item['tokens'] = self.tokens[idx]
        # (number of sentences, the length of the longest sentence, ELMo dimensionality)
        encodings_elmo = self.encodings_elmo[idx]

        return item, encodings_elmo

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.labels)

    def get_unique_labels(self) -> List[str]:
        """Returns a list of unique labels in the dataset."""
        unique_labels = set()
        for labels in self.labels:
            unique_labels.update(labels)
        return list(unique_labels)
def main(args):
    # PATH_TO_ELMO = args.model_name
    PATH_TO_ELMO = "norelmo30"
    elmo_model = ElmoModel()
    elmo_model.load(PATH_TO_ELMO) # ,full=True
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name
    )
    tokens, labels = get_data(
        args.path2data
    )
    tokens, labels  = tokens[:20],labels[:20]
    train_labels, test_labels, train_tokens, test_tokens = datasplit(
        tokens, labels, test_size=0.2
    )
    train_encodings, train_alligned_labels,train_encodings_elmo = get_encodings_and_labels_elmo(train_tokens, train_labels, tokenizer,elmo_model)
    train_encodings_elmo = train_encodings_elmo.astype(np.float32)
    train_encodings_elmo = torch.from_numpy(train_encodings_elmo).float()
    train_dataset = NERDataset_ForElmo(train_encodings, train_alligned_labels,train_tokens,train_encodings_elmo)

    test_encodings, test_alligned_labels,test_encodings_elmo  = get_encodings_and_labels_elmo(test_tokens, test_labels, tokenizer,elmo_model)
    test_encodings_elmo = test_encodings_elmo.astype(np.float32)
    test_encodings_elmo = torch.from_numpy(test_encodings_elmo).float()
    test_dataset = NERDataset_ForElmo(test_encodings, test_alligned_labels,test_tokens,test_encodings_elmo)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # len(train_dataset.get_unique_labels())
    model = NorbertELMOForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=18,tokenizer=tokenizer
    )
    freeze_model = False
    lr = 6e-5
    print("lr", lr, "freeze_model", freeze_model)
    if freeze_model:
        for param in model.base_model.parameters():
            param.requires_grad = False
    else:
        for param in model.base_model.parameters():
            param.requires_grad = True
    optim = AdamW(model.parameters(), lr=5e-5)
    # 768, 384
    # config = AutoConfig.from_pretrained(args.model_name)
    # config.num_labels = 18
    # model = NorBert_LSTM_Model(config,lstm_embed_size=384, num_labels=len(train_dataset.get_unique_labels()))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optim, gamma=0.9)
    model.to(device)
    model.train()

    start_time = time.time()
    model = train(model, optim, train_loader, scheduler, test_loader, args.epochs, device)
    print(f" Elapsed training time: {time.time() - start_time}")
if __name__ == "__main__":
    parser = ArgumentParser()
    # "/fp/projects01/ec30/IN5550/obligatories/3/norne-nb-in5550-train.conllu.gz"
    parser.add_argument(
        "--path2data", type=str, default="norne-nb-in5550-train.conllu.gz")
    parser.add_argument("--epochs", type=int, default=5)
    # PATH_TO_ELMO = '/fp/projects01/ec30/models/norlm/norelmo30'
    parser.add_argument("--model_name", type=str, default="ltg/norbert3-small")
    # parser.add_argument("--model_name", type=str, default="norelmo30")
    args = parser.parse_args()
    main(args)