"""Resources: https://huggingface.co/transformers/v3.2.0/custom_datasets.html"""


from typing import List, Tuple
import torch


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels) -> None:
        """Dataset class for NER token classification task. Returns aligned input ids and labels.

        Args:
            path2data (str): path to conllu file
            tokenizer : tokenizer from transformers, must be fast. 
        """
        super().__init__()
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx : int) -> Tuple[List[int], List[int]]:
        """Returns the input ids and the labels for the given index."""
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.labels)
    
    def get_unique_labels(self) -> List[str]:
        """Returns a list of unique labels in the dataset."""
        unique_labels = set()
        for labels in self.labels:
            unique_labels.update(labels)
        return list(unique_labels)
    

if __name__ == "__main__":
    from data_utils import get_data
    from transformers import AutoTokenizer
    from get_encodings import get_encodings_and_labels
    

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokens, labels = get_data(
        "/Users/magnusnytun/Documents/git/in5550/obligatory3/data/norne-nb-in5550-train.conllu.gz"
    )
    encodings, alligned_labels = get_encodings_and_labels(tokens, labels, tokenizer)
    dataset = NERDataset(encodings, alligned_labels)
    print(dataset[:10])
