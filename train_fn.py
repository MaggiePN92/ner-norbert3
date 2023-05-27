from torch.nn import functional as F
from eval_func import evaluate_seqs
from tqdm import tqdm
from utils.eary_stopping import EarlyStopping
import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def train(
        model, optim, train_loader, scheduler,
        test_loader, n_epochs=5, device="cuda"
):
    accum_loss = []
    model.to(device)
    early_stopper = EarlyStopping()
    prev_macro_f1 = 0.0
    best_clf = -100
    logger.info(f"Starting training with {n_epochs} epochs.")
    for epoch in tqdm(range(n_epochs)):
        model.train()
        accum_loss = 0
        for batch in tqdm(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # Model Outputs: [0] Loss, [1] Logits, [2] Hidden States, [3] Attention Weights
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
    return model