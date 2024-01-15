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

ROOT = get_var_envs()['root']
params = get_emb_params()
logger = setup_logging("Words -- Embeddings", "embeddings")


def get_sentences(data, feature):
    sentences = data[feature].to_list()
    sentences = [gensim.utils.simple_preprocess(text) for text in sentences]
    return sentences


def train(sentences):
    logger.info("Training Word2Vec model...")
    model = gensim.models.Word2Vec(min_count=params["w2v_min_count"], window=params["w2v_window"],
                                   vector_size=params["w2v_size"], seed=42, workers=1)

    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=params['w2v_epochs'])
    model_vectors = model.wv
    model_words = model_vectors.index_to_key
    logger.info("Vocab size: %i" % len(model_words))
    logger.info("Model Word2Vec trained !!!")
    return model, model_words, model_vectors


# Préparation des sentences (tokenization)
def sentences_tokenizer(sentences):
    logger.info("Fit Tokenizer ...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    x_sentences = pad_sequences(tokenizer.texts_to_sequences(sentences),
                                maxlen=params['maxlen'],
                                padding='post')
    num_words = len(tokenizer.word_index) + 1
    logger.info("Number of unique words: %i" % num_words)
    return tokenizer, x_sentences


def embeddings_matrix_func(sentences, tokenizer):
    # Création de la matrice d'embedding
    logger.info("Create Embedding matrix ...")
    mod, w2v_words, model_vectors = train(sentences)
    # tokenizer = sentences_tokenizer(sentences)
    w2v_size = 300
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, w2v_size))
    i = 0
    j = 0

    for word, idx in word_index.items():
        i += 1
        if word in w2v_words:
            j += 1
            embedding_vector = model_vectors[word]
            if embedding_vector is not None:
                embedding_matrix[idx] = model_vectors[word]

    word_rate = np.round(j / i, 4)
    logger.info("Word embedding rate : ", word_rate)
    logger.info("Embedding matrix: %s" % str(embedding_matrix.shape))
    return embedding_matrix, vocab_size


def build_embedding_model(x_sentences, embedding_matrix, vocab_size):
    input = Input(shape=(len(x_sentences), params['maxlen']), dtype='float64')
    word_input = Input(shape=(params['maxlen'],), dtype='float64')
    word_embedding = Embedding(input_dim=vocab_size,
                               output_dim=params['w2v_size'],
                               weights=[embedding_matrix],
                               input_length=params['maxlen'])(word_input)
    word_vec = GlobalAveragePooling1D()(word_embedding)
    embed_model = Model([word_input], word_vec)

    embed_model.summary()
    return embed_model


def load_data():
    data_path = os.path.join(ROOT, f"outputs/data/")
    list_of_files = glob.glob(data_path + "data_clean_*")
    latest_file = max(list_of_files, key=os.path.getctime)
    data = pd.read_csv(latest_file)
    return data


def build_embeddings(datas, feature):
    sentences = get_sentences(datas, feature)
    tokenizer, x_sentences = sentences_tokenizer(sentences)
    embeddings_matrix, vocab_size = embeddings_matrix_func(sentences, tokenizer)
    embed_model = build_embedding_model(x_sentences, embeddings_matrix, vocab_size)
    embeddings = embed_model.predict(x_sentences)
    return embeddings, embed_model, embeddings_matrix


