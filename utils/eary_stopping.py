class EarlyStopping:
    def __init__(
        self, 
        patience : int = 10, 
        min_delta : float = 0.005
    ):
        """EarlyStopping is used to stop training when the model stops
        improving on the validation set. This can reduce overfitting 
        and lead to a better performing model. 

        This code is an adaptet version of the code found in this 
        stackoverflow question: 
        
        https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

        Args:
            patience (int, optional): how many epochs to run if model is
            not improving. Defaults to 15.
            min_delta (float, optional): minimum performance improvement that
            counts. Anything below this will increase the counter. Defaults 
            to 0.001.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.curr_best_macro_f1 = 0

    def early_stop(self, macro_f1 : float) -> bool:
        """Checks if macro_f1 score has improved more than min_delta. If
        macro_f1 has not improved by atleast min_delta in the last patience 
        iterations False is returned, else True. 

        Args:
            macro_f1 (float): f1-score

        Returns:
            bool: if score improved by min_delta the last patience iterations
        """
        if macro_f1 > self.curr_best_macro_f1:
            self.curr_best_macro_f1 = macro_f1
            self.counter = 0
        elif macro_f1 < (self.curr_best_macro_f1 - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
