import torch
from word_embedding.load_embedding import load_embedding


class PretrainedEmbedding:
  def __init__(
    self, 
    path : str
  ) -> None:
    """Loads and implements some helper methods for gensim word
    embeddings. 

    Args:
        path (str): path to embedding that should be loaded. 
    """
    self.emb_model = load_embedding(path)

    print(f"Embedding loaded with vector size = {self.emb_model.vector_size}.")

    self.emb_model["[UNK]"] = torch.tensor(
      self.emb_model.vectors
    ).mean(dim = 0).numpy()

    self.emb_model["[PAD]"] = torch.zeros(
      self.emb_model.vector_size
    ).numpy()

  def get_index(self, token, default):
    return self.emb_model.get_index(token, default=default)

  def get_pad_idx(self):
    return self.emb_model.get_index("[PAD]")
  
  def get_unk_idx(self):
    return self.emb_model.get_index("[UNK]")

  def get_vectors(self):
    return torch.from_numpy(self.emb_model.vectors)
