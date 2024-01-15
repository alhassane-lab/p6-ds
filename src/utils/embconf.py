import glob
import os

import gensim
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.utils import get_var_envs, get_emb_params, setup_logging

class WordEmbeddings:
    def __init__(self):
        self.ROOT = get_var_envs()['root']
        self.params = get_emb_params()
        self.logger = setup_logging("Words -- Embeddings", "embeddings")
        self.model = None
        self.model_words = None
        self.model_vectors = None
        self.tokenizer = None
        self.x_sentences = None
        self.embedding_matrix = None
        self.vocab_size = None
        self.embed_model = None

    def get_sentences(self, data: pd.DataFrame, feature: str) -> list:
        sentences = data[feature].to_list()
        return [gensim.utils.simple_preprocess(text) for text in sentences]

    def train_word2vec_model(self, sentences: list) -> None:
        self.logger.info("Training Word2Vec model...")
        self.model = gensim.models.Word2Vec(
            min_count=self.params["w2v_min_count"],
            window=self.params["w2v_window"],
            vector_size=self.params["w2v_size"],
            seed=42,
            workers=1
        )

        self.model.build_vocab(sentences)
        self.model.train(sentences, total_examples=self.model.corpus_count, epochs=self.params['w2v_epochs'])
        self.model_vectors = self.model.wv
        self.model_words = self.model_vectors.index_to_key
        self.logger.info("Vocab size: %i" % len(self.model_words))
        self.logger.info("Model Word2Vec trained !!!")

    def sentences_tokenizer(self, sentences: list) -> None:
        self.logger.info("Fit Tokenizer ...")
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(sentences)
        self.x_sentences = pad_sequences(self.tokenizer.texts_to_sequences(sentences),
                                         maxlen=self.params['maxlen'],
                                         padding='post')
        num_words = len(self.tokenizer.word_index) + 1
        self.logger.info("Number of unique words: %i" % num_words)

    def embeddings_matrix_func(self, sentences: list) -> None:
        self.logger.info("Create Embedding matrix ...")
        self.train_word2vec_model(sentences)
        w2v_size = 300
        word_index = self.tokenizer.word_index
        self.vocab_size = len(word_index) + 1
        self.embedding_matrix = np.zeros((self.vocab_size, w2v_size))
        i = 0
        j = 0

        for word, idx in word_index.items():
            i += 1
            if word in self.model_words:
                j += 1
                embedding_vector = self.model_vectors[word]
                if embedding_vector is not None:
                    self.embedding_matrix[idx] = self.model_vectors[word]

        word_rate = np.round(j / i, 4)
        self.logger.info("Word embedding rate: %s" % str(word_rate))
        self.logger.info("Embedding matrix: %s" % str(self.embedding_matrix.shape))

    def build_embedding_model(self) -> None:
        self.logger.info("Build Embedding model ...")
        input_layer = Input(shape=(len(self.x_sentences), self.params['maxlen']), dtype='float64')
        word_input = Input(shape=(self.params['maxlen'],), dtype='float64')
        word_embedding = Embedding(input_dim=self.vocab_size,
                                   output_dim=self.params['w2v_size'],
                                   weights=[self.embedding_matrix],
                                   input_length=self.params['maxlen'])(word_input)
        word_vec = GlobalAveragePooling1D()(word_embedding)
        self.embed_model = Model([word_input], word_vec)
        self.embed_model.summary()

    def load_data(self) -> pd.DataFrame:
        data_path = os.path.join(self.ROOT, "outputs/data/")
        list_of_files = glob.glob(data_path + "data_clean_*")
        latest_file = max(list_of_files, key=os.path.getctime)
        data = pd.read_csv(latest_file)
        return data

    def perform_embeddings(self, data: pd.DataFrame, feature: str) -> tuple:
        sentences = self.get_sentences(data, feature)
        self.sentences_tokenizer(sentences)
        self.embeddings_matrix_func(sentences)
        self.build_embedding_model()
        embeddings = self.embed_model.predict(self.x_sentences)
        return embeddings, self.embed_model, self.embedding_matrix


# Example usage
if __name__ == "__main__":
    embeddings_instance = WordEmbeddings()
    data = embeddings_instance.load_data()
    feature_to_embed = "your_feature_name"
    embeddings_result = embeddings_instance.perform_embeddings(data, feature_to_embed)
