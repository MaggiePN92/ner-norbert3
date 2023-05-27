"""
https://huggingface.co/docs/transformers/tasks/token_classification
https://huggingface.co/spaces/evaluate-metric/seqeval
"""
# from utils.ner_eval import Evaluator
from utils.labels2int import Label2Int
import torch
from seqeval.metrics import classification_report


@torch.no_grad()
def evaluate_seqs(model, dataloader, device, label2int = Label2Int()):
    model.eval()

    y_preds = []
    y_trues = []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
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
    return clf_rep_str, clf_rep_dict

def f1(precision, recall):
    score = 2 * (precision * recall) / (precision + recall + 1e-6)
    return score
