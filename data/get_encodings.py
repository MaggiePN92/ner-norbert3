import pathlib
from typing import List, Tuple
import json


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
        self, labels : List[str], word_ids : List[str]
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


def get_encodings_and_labels(
        tokens : List[str], labels : List[str], tokenizer
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
        truncation=True, 
        is_split_into_words=True,
        padding=True,
        max_length=200,
    )

    alligned_labels = []
    for i, label in enumerate(labels):
        word_ids = encodings.word_ids(i)
        alligned_labels.append(align_labels_with_tokens(label, word_ids))

    return encodings, alligned_labels


if __name__ == "__main__":
    from data_utils import get_data
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokens, labels = get_data(
        "/Users/magnusnytun/Documents/git/in5550/obligatory3/data/norne-nb-in5550-train.conllu.gz"
    )
    encodings, alligned_labels = get_encodings_and_labels(tokens, labels, tokenizer)


    print(encodings[0].ids)
    print(alligned_labels[0])

    assert len(encodings[0].ids) == len(alligned_labels[0])
