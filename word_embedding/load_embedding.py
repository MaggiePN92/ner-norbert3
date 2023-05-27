# taken from notebook for lab 4
import gensim

def load_embedding(modelfile):
    # Detect the model format by its extension:
    # Binary word2vec format:
    if modelfile.endswith(".bin.gz") or modelfile.endswith(".bin"):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=True, unicode_errors="replace"
        )
    # Text word2vec format:
    elif (
            modelfile.endswith(".txt.gz")
            or modelfile.endswith(".txt")
            or modelfile.endswith(".vec.gz")
            or modelfile.endswith(".vec")
    ):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=False, unicode_errors="replace"
        )
    else:  # Native Gensim format?
        emb_model = gensim.models.KeyedVectors.load(modelfile)
        #  If you intend to train the model further:
        # emb_model = gensim.models.Word2Vec.load(embeddings_file)
    return emb_model

if __name__ == "__main__":
    pass