""" Module for data preprocessing """
import spacy
from typing import Union
from datetime import datetime
import click
import pandas as pd
from spacy.tokens import Doc
from src.utils import setup_logging
from unidecode import unidecode
from nltk.stem import SnowballStemmer
from src.pipelines.text.integrator import DataIntegrator

TODAY = datetime.today().strftime("%Y%m%d")
logger = setup_logging("P6-DS-Preprocessing", "processing")


class TextTransformer(DataIntegrator):
    """ Class for data transforming and cleaning"""

    def __init__(self) -> None:
        super().__init__()
        self.nlp = spacy.load("en_core_web_lg")
        self.data = self.load_csv(self.data_origins)

    def tokenize_text(self, text: str) -> Doc | Doc | Doc:
        return self.nlp(text)

    def create_corpus(self, categ: bool = True) -> Union[str, list[str]]:
        if categ:
            return " ".join(self.data[self.data['category_1'] == categ]['description'].values)
        return " ".join(self.data['description'].values)

    def count_word_in_category(self, word: str, data: any, category_col: str) -> int:
        category_corpus_list = [self.create_corpus(data, categ) for categ in data[category_col].unique()]
        return sum(word in categ_corpus.lower() for categ_corpus in category_corpus_list)

    #

    def viz_wordcloud_per_category(
            self, data: pd.DataFrame, category_col: str, extra_words: set[str]
    ) -> None:
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8, 10))
        axes = axes.flatten()

        def get_wordcloud(my_corpus) -> WordCloud:
            wordcloud = WordCloud(
                background_color="white", collocations=False, max_words=100
            ).generate(" ".join(my_corpus))
            return wordcloud

        for i, categ in enumerate(data[category_col].unique()):
            corpus_categ = self.create_corpus()
            cleaned_corpus = self.clean_text(corpus_categ)
            wordcloud_categ = get_wordcloud(cleaned_corpus)
            axes[i].imshow(wordcloud_categ, interpolation="bilinear")
            axes[i].set_title(f"{i + 1} - {categ}")
            axes[i].axis("off")
        file_name = f"plots/word_clouds.png"
        plt.tight_layout()
        plt.savefig(self.outputs_dir + file_name)
        plt.show()

    def get_extra_stop_words(self) -> set[str]:
        logger.info(f"Defaults stopwords number <-->  {len(self.stopwords)}")
        extra_words: set[str] = set()

        raw_corpus = self.create_corpus(False)
        logger.info(f"Successfully created corpus | Size:{len(raw_corpus)} | unique words:{len(set(raw_corpus))}")

        raw_corpus_cleaned = self.clean_text(raw_corpus, extra_words)
        logger.info(
            f"The corpus is cleaned <---> Size : {len(raw_corpus_cleaned)} <---> Unique_words_count: {len(set(raw_corpus_cleaned))}")
        tmp = pd.Series(raw_corpus_cleaned).value_counts()
        logger.info("Computing unique words ...")
        unique_words = list(set(tmp[tmp == 1].index))
        extra_words.update(unique_words)
        logger.info(f"{len(unique_words)} unique words added to extra stop words!!!")

        top_common_words = [word for word in tmp[tmp == 2].index if
                            self.count_word_in_category(word, self.data, 'category_1') == len(
                                self.data.category.unique())]
        extra_words.update(top_common_words)
        logger.info(f"{len(top_common_words)} top common words added to extra stop words!!!")
        most_frequent_common_words = [word for word in tmp[tmp.values[:50]].index if
                                      self.count_word_in_category(word, self.data, 'category_1') > 4]
        extra_words.update(most_frequent_common_words)
        logger.info(f"{len(most_frequent_common_words)} most frequent common words added in the list.")

        logger.info(f"Total extra words to remove : {len(extra_words)}")
        return extra_words

    def clean_text(self, text: str, extra_words: set[str], use_stemmer: bool = False) -> list[str]:
        # Custom infix patterns
        infix_patterns = list(self.nlp.Defaults.infixes)
        infix_patterns.extend([r"[A-Z][a-z0-9]+", r'\b\w+-\w+\b'])
        infix_regex = spacy.util.compile_infix_regex(infix_patterns)
        self.nlp.tokenizer.infix_finditer = infix_regex.finditer

        doc = self.nlp(text)

        # Stop words
        stopwords = self.nlp.Defaults.stop_words
        stopwords = stopwords.update(extra_words)

        # Stemmer
        stemmer = SnowballStemmer("english") if use_stemmer else None

        text_clean = [
            unidecode(stemmer.stem(token.text.lower())) if use_stemmer else unidecode(token.lemma_.lower())
            for token in doc
            if (
                    not token.is_punct
                    and not token.is_space
                    and not token.is_stop
                    and not token.like_url
                    and not len(token) < 3
                    and not str(token).lower() in stopwords
            )
        ]
        return text_clean

    def full_process_data(self, data: object) -> None:
        extra_words = self.get_extra_stop_words(data)
        data = (data
                .assign(lema=data.description
                        .apply(lambda x: self.clean_text(x, extra_words, False))
                        .apply(lambda x: " ".join(x)))
                .assign(stem=data.description
                        .apply(lambda x: self.clean_text(x, extra_words, True))
                        .apply(lambda x: " ".join(x)))
                )
        data['_len__lema'] = data['lema'].apply(lambda x: len(str(x)))
        data['_len__stem'] = data['stem'].apply(lambda x: len(str(x)))
        logger.info(f"Data Infos : {data.info()}")
        # self.viz_wordcloud_per_category(data, 'category', extra_words)
        file_name = f"data/data_clean_{TODAY}.csv"
        data.to_csv(self.outputs_dir + file_name, index=False)
        logger.info(f"Cleaned data saved to outputs/{file_name}")


@click.command
@click.pass_context
def process_data(ctx: click.Context) -> None:
    """
    Root command.
    """
    preprocessor = DataTransformer()
    preprocessor.extract_features()
    preprocessor.select_features()

    raw_data = preprocessor.get_extra_stop_words()

    print(raw_data)
    # cleaned_data = preprocessor.process_data(raw_data)
    # cleaned_data = perform_eda(cleaned_data)
    # preprocessor.full_process_data(cleaned_data)
