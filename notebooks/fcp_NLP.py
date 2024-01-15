###########################################################################################################################
# Fichier de fonctions du Projet 6 - fct_NLP
###########################################################################################################################

###########################################################################################################################
## 1. Importation des libraires :

import numpy as np
import pandas as pd

## NLTK
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer


###########################################################################################################################
## 2. PRE-TRAITEMENT DES DONNEES TEXTUELLES


def tokens_info(token):
    """
    Renvoie le nombre de tokens total et le nombre de tokens uniques

    Args:
        token (list(str)): listes de chaines de caractères
    """
    print(f"nb tokens {len(token)}, nb tokens uniques {len(set(token))}")


def clean_process_text(doc, stop_words=None, min_len_word=None, alpha='n', lemm_or_stemm=None, join=False):
    """
    Nettoie le document avec différentes options à définir.

    Args:
        doc (str): document à nettoyer.
        stop_words (list(str), optional): liste de mots à éliminer. Defaults to None.
        min_len_word (int, optional): Nombre de caractère minimum autorisé par mot. Defaults to None.
        alpha (str, optional): Indication de la suppression ou non des tokens numérique. Defaults to 'n'
        lemm_or_stemm (str, optional): choix de Lemmatizer ou Stemmer. Defaults to None.
        join (bool, optional): renvoyer une liste de tokens (si faux) ou une chaine de caractère . Defaults to False.

    Returns:
        document nettoyé : liste de token (si join est Flase) ou chaine de caractère
    """

    # 1. Initialisation de la liste des stop_words si non remplie :
    if not stop_words:
        stop_words = []

    # 2. Passage du document en minuscule et suppression des espaces en début et fin :
    doc = doc.lower().strip()

    # 3. Tokenization :
    tokenizer = RegexpTokenizer(r"\w+")
    tokens_list = tokenizer.tokenize(doc)

    # 4. Elimination des stop_words :
    cleaned_tokens_list = [w for w in tokens_list if w not in stop_words]

    # 5. Suppression des mots avec moins de min_len_word caractères :
    if min_len_word:
        cleaned_tokens_list = [w for w in cleaned_tokens_list if len(w) >= min_len_word]

    # 6. Suppression des tokens non alphabétiques :
    if alpha == 'y':
        cleaned_tokens_list = [w for w in cleaned_tokens_list if w.isalpha()]

    # 7. Application du Stemmer ou du Lemmatizer :
    if lemm_or_stemm == 'lem':
        trans = WordNetLemmatizer()
        trans_text = [trans.lemmatize(w) for w in cleaned_tokens_list]
    elif lemm_or_stemm == 'stem':
        trans = PorterStemmer()
        trans_text = [trans.stem(w) for w in cleaned_tokens_list]
    else:
        trans_text = cleaned_tokens_list

    # 8. Regroupement des mots dans la même chaine de caractère :
    if join:
        cleaned_description = ' '.join(trans_text)
    else:
        cleaned_description = trans_text

    return cleaned_description


###########################################################################################################################
## 3. NLP


def mean_embending_w2v_df(descriptions, word2vec_model):
    """
    Calcule les embeddings de descriptions à partir de descriptions tokenisées et d'un modèle Word2Vec, et retourne un DataFrame.

    Args:
        sentences (list of list): Une liste de listes de mots où chaque liste représente une description.
        word2vec_model (Word2Vec): Un modèle Word2Vec préalablement entraîné.

    Returns:
        pd.DataFrame: Un DataFrame contenant les embeddings de descriptions.
    """

    # 1. Initialisation de la liste des embeddings moyens :
    embeddings_mean = []

    # 2.Calcul de l'embedding moyen de chaque description :
    for description in descriptions:
        # Vérification de la présence des mot dans le vocabulaire :
        mots_dans_le_vocabulaire = [mot for mot in description if mot in word2vec_model.wv]

        if mots_dans_le_vocabulaire:
            description_embedding = np.mean([word2vec_model.wv[mot] for mot in mots_dans_le_vocabulaire], axis=0)
            embeddings_mean.append(description_embedding)
        else:
            # Si la description est vide, ajout d'un vecteur nul :
            embeddings_mean.append(np.zeros(word2vec_model.vector_size))

    # 3. Créez un DataFrame à partir de la liste d'embeddings de descriptions
    df = pd.DataFrame(embeddings_mean, columns=[f"embedding_{i}" for i in range(word2vec_model.vector_size)])

    return df

