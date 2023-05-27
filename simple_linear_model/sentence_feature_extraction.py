import numpy as np


class FeatureExtractionBaseline:
    def __init__(self, embedding, label2int) -> None:
        self.embedding = embedding
        self.label2int = label2int

    def feature_extraction_token_level(self, sentence, i, k = 1):
        if i < k - 1:
            raise ValueError("i must be at least k - 1")
        
        features = self.get_word_embedding(sentence[i]).squeeze()

        for j in range(1, k + 1):
            left_token  = self.get_word_embedding(sentence[i - j])
            right_token = self.get_word_embedding(sentence[i + j])
            features = np.concatenate(
                (features, left_token.squeeze(), right_token.squeeze())
            )
        return features

    def get_word_embedding(self, token):
        try:
            num_repr = self.embedding.emb_model[token]
        except KeyError:
            num_repr = self.embedding.emb_model["[UNK]"]
        return num_repr

    def pad_sentence(self, sentence, k = 2):
        prefix_tokens = ["<s>"] * k
        suffix_tokens = ["</s>"] * k
        padded_sentence = prefix_tokens + sentence + suffix_tokens
        return padded_sentence

    def feature_extraction(self, tokens, labels, k = 2):
        """Extracts features for linear model. It returns nested lists like 
        [([features for tokens], [labels])]. [features for tokens] and [labels]
        have equal lenghts. This means that the sentence structure is preserved 
        in the feature extraction process. 

        Args:
            tokens (List[List[str]]): Nested list where each list is a sentence
            labels (List[List[str]]): Nested list where each list contains the labels for each token
            k (int, optional): Size if context window. Defaults to 2.

        Returns:
            List[Tuple[List[np.array], List[int]]]: List of list where each list is the features for a sentence.
        """
        data = []
        

        for sentence, sentence_labels in zip(tokens, labels):
            padded_sentence = self.pad_sentence(sentence, k = k)

            features_sentence_level = []    
            for i in range(k, len(padded_sentence) - k):
                features = self.feature_extraction_token_level(padded_sentence, i, k = k)
                features_sentence_level.append(features)

            data.append((np.stack(features_sentence_level, axis=0), [self.label2int(l) for l in sentence_labels]))
        return data


if __name__ == "__main__":
    """Tests the feature extraction baseline."""
    
    labels = [["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], 
              ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
              ["0","1","2"]]
    
    tokens = [["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"], 
              ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
              ["a", "b", "c"]]

    def mock_label2int(label):
        return label

    class MockEmbedding:
        def __init__(self) -> None:
            self.emb_model = {
                "a": np.array([1, 1]), "b": np.array([2, 2]), 
                "c": np.array([3, 3]), "d": np.array([4, 4]), 
                "e": np.array([5, 5]), "f": np.array([6, 6]), 
                "g": np.array([7, 7]), "h": np.array([8, 8]), 
                "i": np.array([9, 9]), "j": np.array([10, 10]),
                "[UNK]" : np.array([0, 0])
            }

    embedding = MockEmbedding()

    feature_extractor = FeatureExtractionBaseline(embedding, mock_label2int)
    data = feature_extractor.feature_extraction(tokens, labels, k = 1)
    
    sentences = data[2][0]
    labels = data[2][1]

    print(sentences)
    print(labels)
    print(sentences.shape)
    print(len(labels))

    print(data)
    