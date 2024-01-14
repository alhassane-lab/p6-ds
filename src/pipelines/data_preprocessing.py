"""
Module for text data exploration, cleaning, visualization, feature extraction. 
"""
import spacy
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.pipelines.eda import (univariate, bivariate, anova, outliers)
from src.utils import setup_logging, get_var_envs
from unidecode import unidecode
from wordcloud import WordCloud


logger = setup_logging("Preprocessing")
var_envs = get_var_envs()


class DataProcessor:
    """
    This class contains functions for data exploration, cleaning and preprocessing.
    """

    def __init__(self) -> None:
        """
        Initializes the DataProcessor class.
        """
        logger.info("========== Initializing ==========")
        print(Path(".").resolve())
        root_dir = var_envs['root']
        self.data_dir = os.path.join(root_dir, 'data/')
        logger.info("Data directory is loaded !!!")
        self.outputs_dir = os.path.join(root_dir, 'outputs/')
        logger.info("Outputs directory is loaded !!!")
        model = "en_core_web_sm"
        self.nlp = spacy.load(model)
        logger.info(f"Spacy model {model} is loaded !!!")

    def load_data(self) -> pd.DataFrame:
        """
        import data from local path
        """
        logger.info("========== Loading Data... ==========")
        data = pd.read_csv(self.data_dir + var_envs['file'])
        logger.info(f"Features: {list(data.columns)}")
        logger.info(f"Raw data shape: {data.shape}")
        return data

    @staticmethod
    def process_data(data):
        logger.info("========== Processing Phase 1 ==========")
        tmp = data.description.str.len()
        data = (data
                .assign(category=data.product_category_tree.str.split(">>").str[0].str.strip('["'))
                .assign(_len_description=tmp)
                )[['uniq_id', 'description', 'category', '_len_description']]
        logger.info(" ---> New feature : category ==> Extracted from product_category_tree")
        logger.info(" ---> New feature : _len_description  ==> Computed from description length")
        logger.info("Processed Data Features: {}".format(list(data.columns)))
        logger.info("Processed Data shape: {}".format(data.shape))
        return data

    def perform_eda(self, data):
        numerical = '_len_description'
        categorical = 'category'
        eda_results_path = self.outputs_dir
        for folder in ['plots/', 'data/']:
            my_folder = eda_results_path + folder
            if not os.path.exists(my_folder):
                os.makedirs(my_folder)
        univariate(data, numerical, eda_results_path)
        bivariate(data, (numerical, categorical), eda_results_path)
        anova(data, (numerical, [categorical]))
        data, outs = outliers(data, numerical, eda_results_path)
        logger.info(f"{len(outs)} outliers cleaned...")
        logger.info("Processed Data shape: {}".format(data.shape))
        return data

    def clean_text(
            self, text: str, force_is_alpha: bool, extra_words: list[str]
    ) -> list[str]:

        # Add custom infix patterns for tokenization
        infix_patterns = list(self.nlp.Defaults.infixes)
        infix_patterns.extend([r"[A-Z][a-z0-9]+", r'\b\w+-\w+\b'])
        infix_regex = spacy.util.compile_infix_regex(infix_patterns)
        self.nlp.tokenizer.infix_finditer = infix_regex.finditer

        doc = self.nlp(text)
        text_clean = [
            unidecode(token.lemma_.lower())
            for token in doc
            if (
                    not token.is_punct
                    and not token.is_space
                    and not token.like_url
                    and not token.is_stop
                    and not len(token) < 3
                    and not str(token).lower() in extra_words
            )
        ]

        if force_is_alpha:
            alpha_tokens = [w for w in text_clean if w.isalpha()]
        else:
            alpha_tokens = text_clean
        return alpha_tokens

    @staticmethod
    def create_corpus(data: object, categ: str = None) -> str:
        if categ:
            return " ".join(data[data['category'] == categ]['description'].values)
        return " ".join(data['description'].values)

    def count_word_in_category(self, word: str, data: any, category_col: str) -> int:
        category_corpus_list = [self.create_corpus(data, categ) for categ in data[category_col].unique()]
        return sum(word in categ_corpus.lower() for categ_corpus in category_corpus_list)

    def viz_wordcloud_per_category(
            self, data: object, category_col: str, extra_words: list[str]
    ) -> None:
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8, 10))
        axes = axes.flatten()

        def get_wordcloud(my_corpus) -> WordCloud:
            wordcloud = WordCloud(
                background_color="white", collocations=False, max_words=100
            ).generate(" ".join(my_corpus))
            return wordcloud

        for i, categ in enumerate(data[category_col].unique()):
            wordcloud_categ = get_wordcloud(
                self.clean_text(self.create_corpus(data, categ), False, extra_words=extra_words)
            )
            axes[i].imshow(wordcloud_categ, interpolation="bilinear")
            axes[i].set_title(f"{i + 1} - {categ}")
            axes[i].axis("off")
        file_name = f"plots/word_clouds.png"
        plt.tight_layout()
        plt.savefig(self.outputs_dir + file_name)
        plt.show()

    def get_extra_stop_words(self, data):
        logger.info("========== Extra stop words  ==========")
        extra_words = []
        raw_corpus = self.create_corpus(data)
        raw_corpus_cleaned = self.clean_text(raw_corpus, False, extra_words)
        tmp = pd.Series(raw_corpus_cleaned).value_counts()
        unique_words = list(set(tmp[tmp == 1].index))
        extra_words.extend(unique_words)
        logger.info(f"{len(unique_words)} unique words added to extra stop words!!!")
        # List of common words to all categories simultaneously.
        top_common_words = [word for word in tmp.index if
                            self.count_word_in_category(word, data, 'category') >= 4]

        extra_words.extend(top_common_words)
        logger.info(f"{len(top_common_words)} top common words added to extra stop words!!!")
        return extra_words

    def full_process_data(self, data):
        extra_words = self.get_extra_stop_words(data)
        data = data.assign(text=data.description.apply(
            lambda x: self.clean_text(x, False, extra_words)).apply(
            lambda x: " ".join(x)))
        self.viz_wordcloud_per_category(data, 'category', extra_words=extra_words)
        file_name = f"data/data_clean.csv"
        data.to_csv(self.outputs_dir + file_name, index=False)
        logger.info(f"Cleaned data saved to outputs/{file_name}")
        return data
