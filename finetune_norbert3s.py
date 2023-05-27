import torch
import time
import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from data.dataset import NERDataset
from data.data_utils import get_data, datasplit
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AdamW
from data.get_encodings import get_encodings_and_labels
from torch.utils.data import DataLoader
from utils.modeling_norbert import NorbertForTokenClassification
from eval_func import evaluate_seqs
from utils.eary_stopping import EarlyStopping


logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def main(global_args, hp_args):
    tokenizer = AutoTokenizer.from_pretrained(
        global_args.model_name
    )
    tokens, labels = get_data(
        global_args.path2data
    )
    train_labels, test_labels, train_tokens, test_tokens = datasplit(
        tokens, labels, test_size=0.2
    )
    train_encodings, train_alligned_labels = get_encodings_and_labels(train_tokens, train_labels, tokenizer)
    train_dataset = NERDataset(train_encodings, train_alligned_labels)

    test_encodings, test_alligned_labels = get_encodings_and_labels(test_tokens, test_labels, tokenizer)
    test_dataset = NERDataset(test_encodings, test_alligned_labels)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


    prev_best_f1 = 0
    for model_args in hp_args:
        logger.info(f"Training model with hyperparameters: {model_args}")

        model = NorbertForTokenClassification.from_pretrained(
            global_args.model_name, 
            num_labels=len(train_dataset.get_unique_labels()) - 1,
            num_hidden_layers=model_args.num_hidden_layers,
            hidden_dropout_prob=model_args.hidden_dropout_prob
        )
        
        if not model_args.finetune_transformer:
            for param in model.base_model.transformer.parameters():
                param.requires_grad = False

        if not model_args.finetune_embedding:
            for param in model.base_model.embedding.parameters():
                param.requires_grad = False
        
        optim = AdamW(
            model.parameters(), 
            lr=model_args.lr
        )
        n_batches = len(train_loader)
        
        if model_args.lr_warmup:
            n_training_steps = n_batches * global_args.epochs
            n_warmup_steps = int(0.1 * n_training_steps)
            scheduler = get_linear_schedule_with_warmup(optim, n_warmup_steps, n_training_steps)
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)

        model.to(device)
        model.train()

        start_time = time.time()
        model, macro_f1 = train(model, optim, train_loader, scheduler, test_loader, global_args.epochs, device)
        if macro_f1 > prev_best_f1:
            prev_best_f1 = macro_f1
            best_model = model
            best_args = model_args

        logger.info(f" Elapsed training time: {time.time() - start_time}")


    logger.info(f"Best model: {best_args}")
    logger.info(f"Best macro f1: {prev_best_f1}")
    torch.save(best_model.state_dict(), "norbert3_large")


def train(
    model, optim, train_loader, scheduler, 
    test_loader, n_epochs = 5, device = "cuda"
):
    accum_loss = []
    model.to(device)
    early_stopper = EarlyStopping()
    prev_macro_f1 = 0.0
    
    logger.info(f"Starting training with {n_epochs} epochs.")
    for epoch in range(n_epochs):
        accum_loss = 0
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
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
    model.load_state_dict(best_state)
    return model, prev_macro_f1


@dataclass
class SearchArgs:
    num_hidden_layers : int = 12
    hidden_dropout_prob : float = 0.1
    finetune_embedding : bool = True
    finetune_transformer : bool = True
    lr : float = 5e-5
    lr_warmup : bool = False


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path2data", type=str, default="/fp/projects01/ec30/IN5550/obligatories/3/norne-nb-in5550-train.conllu.gz")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model_name", type=str, default="ltg/norbert3-large")
    global_args = parser.parse_args()

    hp_args = [
        SearchArgs(12, 0.1, False, False, 5e-5, False),
        SearchArgs(12, 0.1, True, True, 5e-5, True),
        SearchArgs(12, 0.1, False, False, 5e-5, True)
    ]
    
    main(global_args, hp_args)
