import time
from train_fn import train
from data.dataset import NERDataset
from data.data_utils import get_data, datasplit
from transformers import AutoTokenizer
from data.get_encodings import get_encodings_and_labels
from torch.utils.data import DataLoader
from transformers import AdamW
from utils.modeling_norbert import NorbertForTokenClassification,NorbertLSTMForTokenClassification
import torch
# from modeling_norbert import NorbertForMaskedLM
from utils.modeling_norbert import NorbertModel
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoConfig, AutoModel
from tqdm import tqdm
# from eval_func import evaluate_seqs
import transformers
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
import logging
# from seqeval.metrics import f1_score, precision_score, recall_score
from utils.eary_stopping import EarlyStopping
from utils.labels2int import Label2Int
import torch
from seqeval.metrics import classification_report

logger = logging.getLogger(__name__)
# def train(
#         model, optim, train_loader, scheduler,
#         test_loader, n_epochs=5, device="cuda"
# ):
#     accum_loss = []
#     model.to(device)
#     early_stopper = EarlyStopping()
#     prev_macro_f1 = 0.0
#     best_clf = -100
#     logger.info(f"Starting training with {n_epochs} epochs.")
#     for epoch in tqdm(range(n_epochs)):
#         model.train()
#         accum_loss = 0
#         for batch in tqdm(train_loader):
#             optim.zero_grad()
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
#             # Model Outputs: [0] Loss, [1] Logits, [2] Hidden States, [3] Attention Weights
#             logits = model(input_ids, attention_mask=attention_mask)
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1))
#             accum_loss += loss.item()
#             loss.backward()
#             optim.step()
#
#         print(f"Epoch {epoch} loss: {accum_loss / len(train_loader)}")
#         clf_str, clf_dict = evaluate_seqs(model, test_loader, device)
#         epoch_macro_f1 = clf_dict["macro avg"]["f1-score"]
#         print(f"Epoch {epoch} macro f1: {epoch_macro_f1:.3f}")
#
#         # if current model is better then previous best; save state dict
#
#         if epoch_macro_f1 > prev_macro_f1:
#             best_state = model.state_dict()
#             prev_macro_f1 = epoch_macro_f1
#             best_clf = clf_str
#         # early stopping returns false if model has not improved for k epochs
#         if early_stopper.early_stop(epoch_macro_f1):
#             break
#
#         # adjust learning rate
#         scheduler.step()
#
#     logger.info("Training finished.")
#     logger.info(best_clf)
#     # after training; set model to best version in train loop
#     model.load_state_dict(best_state)
#     return model


# @torch.no_grad()
# def evaluate_seqs(model, dataloader, device, label2int=Label2Int()):
#     model.eval()
#
#     y_preds = []
#     y_trues = []
#     for batch in dataloader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         outputs = model(input_ids, attention_mask=attention_mask)
#         y_preds += outputs.argmax(dim=2).tolist()
#         y_trues += labels.tolist()
#
#     # Create list of predicted labels from logit indeces.
#     true_preds = [
#         [label2int.int2label[p] for (p, l) in zip(y_pred, y_true) if l != -100]
#         for y_pred, y_true in zip(y_preds, y_trues)
#     ]
#     true_labels = [
#         [label2int.int2label[l] for (p, l) in zip(y_pred, y_true) if l != -100]
#         for y_pred, y_true in zip(y_preds, y_trues)
#     ]
#
#     # evalutor = Evaluator(true_labels, true_preds, list(label2int.mapping.keys()))
#     # results, eval_agg = evalutor.evaluate()
#
#     # f1_score = f1_score(true_labels, true_preds, mode="strict")
#     # precision_score = precision_score(true_labels, true_preds, mode="strict")
#     # recall_score = recall_score(true_labels, true_preds, mode="strict")
#     # return f1_score, precision_score, recall_score
#
#     clf_rep_str = classification_report(true_labels, true_preds, mode="strict")
#     clf_rep_dict = classification_report(true_labels, true_preds, mode="strict", output_dict=True)
#     return clf_rep_str, clf_rep_dict


# def f1(precision, recall):
#     score = 2 * (precision * recall) / (precision + recall + 1e-6)
#     return score

# class NorBert_LSTM_Model(nn.Module):
#     def __init__(self, config,lstm_embed_size, num_labels):
#         super().__init__()
#         # self._bert = transformers.BertModel.from_pretrained(norbert)
#         # self._bert = AutoModel.from_config(norbert)
#         self.num_labels = num_labels
#         # self._bert = NorbertModel.from_pretrained(norbert)
#         self._bert = AutoModel.from_config(config)
#         # self._head = nn.Linear(768, num_labels)
#         self.loss_fn = nn.CrossEntropyLoss()
#         for param in self._bert.parameters():
#            param.requires_grad = False
#         hidden_size = lstm_embed_size // 2
#         layer_norm_eps = 1.0e-7
#         drop_out = 0.5
#         self.lstm = nn.LSTM(
#             input_size=lstm_embed_size,
#             hidden_size=hidden_size,
#             batch_first=True,
#             num_layers=2,
#             dropout=0.5,
#             bidirectional=True
#         )
#         self.nonlinearity = nn.Sequential(
#             nn.LayerNorm(hidden_size, layer_norm_eps, elementwise_affine=False),
#             nn.Linear(lstm_embed_size, hidden_size),
#             nn.GELU(),
#             nn.LayerNorm(hidden_size, layer_norm_eps, elementwise_affine=False),
#             nn.Dropout(drop_out),
#             nn.Linear(hidden_size, num_labels),
#             # nn.GELU(),
#         )
#         # self.classifier = nn.Linear(lstm_embed_size, num_labels)
#     def forward(self, input_ids, attention_mask):
#         # b = self._bert(x)
#         b = self._bert(input_ids, attention_mask).last_hidden_state
#         lstm_output, _ = self.lstm(b)
#         logits = self.nonlinearity(lstm_output)
#         # pooler shape: batch_size, t, bert_dim
#         # pooler = b.last_hidden_state[:, mask].diagonal().permute(2, 0, 1)
#         # loss = self.loss_fn(torch.transpose(x, 1, 2), labels)
#
#         return logits

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name
    )
    tokens, labels = get_data(
        args.path2data
    )
    # tokens, labels = tokens[:100], labels[:100]
    train_labels, test_labels, train_tokens, test_tokens = datasplit(
        tokens, labels, test_size=0.2
    )
    train_encodings, train_alligned_labels = get_encodings_and_labels(train_tokens, train_labels, tokenizer)
    train_dataset = NERDataset(train_encodings, train_alligned_labels)

    test_encodings, test_alligned_labels = get_encodings_and_labels(test_tokens, test_labels, tokenizer)
    test_dataset = NERDataset(test_encodings, test_alligned_labels)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_labels = len(train_dataset.get_unique_labels())
    model = NorbertLSTMForTokenClassification.from_pretrained(
        args.model_name,
        num_labels= num_labels
    )
    freeze_model = False
    lr = 5e-5
    print("lr", lr, "freeze_model", freeze_model)
    for param in model.head.parameters():
        param.requires_grad = True
    if freeze_model:
        for param in model.base_model.transformer.parameters():
            param.requires_grad = False
        for param in model.base_model.embedding.parameters():
            param.requires_grad = False
    else:
        for param in model.base_model.transformer.parameters():
            param.requires_grad = True
        for param in model.base_model.embedding.parameters():
            param.requires_grad = False
    optim = AdamW(model.parameters(), lr=5e-5)
    # 768, 384
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = num_labels
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
    parser.add_argument(
        "--path2data", type=str, default="norne-nb-in5550-train.conllu.gz")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="ltg/norbert3-small")
    args = parser.parse_args()
    main(args)
