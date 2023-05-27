from sklearn.model_selection import train_test_split
import conllu
import gzip
from typing import List, Tuple


def get_token_tag_lists(sent : conllu.models.TokenList) -> Tuple[List[str], List[str]]:
    """Iterates the sentences and returns the tokens and the NER tags.

    Args:
        sent (conllu.models.TokenList): sentence with tokens and NER tags

    Returns:
        Tuple[List[str], List[str]]: Lists of tokens and NER tags
    """
    ner_tags = []
    tokens = []
    
    for i in range(len(sent)):
        tokens.append(sent[i]["form"])
        ner_tags.append(sent[i]["misc"]["name"])
        
    return tokens, ner_tags


def get_data(path2data : str) -> Tuple[List[List[str]], List[List[str]]]:
    """ Reads the data from the path and returns the tokens and the NER tags.

    Args:
        path2data (str): path to the data

    Returns:
        Tuple[List[List[str]], List[List[str]]]: Nested list of tokens and NER tags
    
    """
    with gzip.open(path2data, "rt", encoding="utf-8") as f:
        data = f.read()

    parsed_data = conllu.parse(data)

    nested_ner_tags = []
    nested_tokens = []
    
    for doc in parsed_data:
        tokens, ner_tags = get_token_tag_lists(doc)
        nested_tokens.append(tokens)
        nested_ner_tags.append(ner_tags)
        
    return nested_tokens, nested_ner_tags


def datasplit(tokens, labels, test_size=0.2):
    return train_test_split(labels, tokens, test_size=test_size)
