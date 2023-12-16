
import spacy
from unidecode import unidecode
from nltk.stem.porter import PorterStemmer


def clean_text(text, force_is_alpha: bool = True, extra_words=[], min_len_word: int = 3,) -> list[str]:
    """
    Aims to process a raw text clean it,
    Args:
        text (str): _description_
        min_len_word (int, optional): token minimum lenght. Defaults to 3.
        regex (str, optional): _description_. Defaults to r"@[\w\d_]+".
        force_is_alpha (bool, optional): _description_. Defaults to True.

    Returns:
        list : _description_
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    stemmer = PorterStemmer()

    text_clean = [
        unidecode(token.lemma_.lower())
        for token in doc
        if (
            not token.is_punct
            and not token.is_space
            and not token.like_url
            and not token.is_stop
            and len(token) >= min_len_word
            #and not token._.like_handle
        )
    ]

    #text_clean = [stemmer.stem(w) for w in text_clean]

    if force_is_alpha:
        alpha_tokens = [w for w in text_clean if w.isalpha()]
    else:
        alpha_tokens = text_clean

    final_tokens = [token for token in alpha_tokens if token not in extra_words]

    return final_tokens