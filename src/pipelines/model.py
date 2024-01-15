import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.utils import setup_logging, get_var_envs
from src.utils.embconf import WordEmbeddings
from sklearn import cluster, metrics
from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt
import click
import glob
import os
import time
import json

logger = setup_logging("Feature-Extraction", "extract")
ROOT = get_var_envs()['root']


def load_data():
    """
    Load data from the latest file in the data directory.
    """
    logger.info("========== Loading Data... ==========")
    data_path = os.path.join(ROOT, f"outputs/data/")
    list_of_files = glob.glob(data_path + "data_clean_*")
    latest_file = max(list_of_files, key=os.path.getctime)
    data = pd.read_csv(latest_file)
    logger.info(f"Features: {list(data.columns)}")
    logger.info(f"Raw data shape: {data.shape}")
    return data


def category_metrics(data, target):
    """
    Calculate category metrics.
    """
    logger.info(f"Target variable: {target}")
    l_cat = list(set(data[target]))
    logger.info(f"Target unique values: {l_cat}")
    y_cat_num = [(1 - l_cat.index(data.iloc[i][target])) for i in range(len(data))]
    return l_cat, y_cat_num


def get_features(data, model: object, feature: str):
    """
    Initialize the model and extract features.
    """
    logger.info("========== Initializing Model ... ==========")
    logger.info(f"Feature variable: {feature}")
    features = model.fit_transform(data[feature])
    logger.info("Data fitting ...")
    return features


def ari_fct(features, perplexity, learning_rate, l_cat, y_cat_num):
    """
    Calculate ARI score using TSNE and KMeans clustering.
    """
    num_labels = len(l_cat)
    tsne = manifold.TSNE(n_components=2, perplexity=perplexity, n_iter=2000,
                         init='random', learning_rate=learning_rate, random_state=42)
    x_tsne = tsne.fit_transform(features)

    cls = cluster.KMeans(n_clusters=num_labels, n_init=100, random_state=42)
    cls.fit(x_tsne)
    ari_score = np.round(metrics.adjusted_rand_score(y_cat_num, cls.labels_), 4)
    return ari_score, x_tsne, cls.labels_


def tsne_visu_fct(x_tsne, labels, l_cat, y_cat_num, model) -> None:
    """
    Visualize TSNE results.
    """
    logger.info("========== Data Visualization  ==========")
    plots_dir = os.path.join(ROOT, f"outputs/plots/")
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(121)
    scatter = ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_cat_num, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=l_cat, loc="best", title="Category")
    plt.title('Description per category')

    ax = fig.add_subplot(122)
    scatter = ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=labels, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=set(labels), loc="best", title="Clusters")
    plt.title('Description par clusters')
    plt.savefig(plots_dir + f'tsne_kmeans_{repr(model)}_plot.png')
    logger.info("Visuals saved to : outputs/plots/tsne_kmeans_plot.png")
    plt.show()


def test_model(data: object,
               target: str,
               model: object,
               features,
               tuning_params: tuple[list[str]],
               ) -> object:
    """
    Test the model with different tuning parameters.
    """
    logger.info("========== Modeling ... ==========")
    results_file = os.path.join(ROOT, "outputs/data/results.json")

    with open(results_file, "r") as f:
        results = json.load(f)

    logger.info("Perplexity x learning_rate")
    logger.info(f"Model name: {repr(model)}")
    l_cat, y_cat_num = category_metrics(data, target)

    i = 0
    for perplexity in tuning_params[0]:
        for learning_rate in tuning_params[1]:
            time1 = time.time()
            ari_score, x_tsne, labels = ari_fct(features, perplexity, learning_rate, l_cat,
                                                y_cat_num)
            time2 = np.round(time.time() - time1, 0)
            logger.info(
                f"Ari-Score: {ari_score} | Perplexity:{perplexity} | Training time{time2}")
            row = [ari_score, perplexity, learning_rate]
            results[f"Test_{i} <--> {repr(model)}"] = row
            i += 1

    with open(results_file, "w") as outfile:
        json.dump(results, outfile)


@click.command()
@click.pass_context
def extract_features(ctx: click.Context):
    """
    Extract features using different models and tuning parameters.
    """
    feature = 'lema_desc'
    logger.info("========== Initializing ==========")
    tuning_params = ([20, 30, 40, 50], [100, 200, 300])
    data = load_data()

    features = get_features(data, CountVectorizer(), feature)
    test_model(data, 'category', CountVectorizer(), features, tuning_params)

    features = get_features(data, TfidfVectorizer(), feature)
    test_model(data, 'category', TfidfVectorizer(), features, tuning_params)

    embeddings_instance = WordEmbeddings()
    embeddings_result = embeddings_instance.perform_embeddings(data, feature)
