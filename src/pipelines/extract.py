import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.utils import setup_logging, get_var_envs
from sklearn import cluster, metrics
from sklearn import manifold, decomposition
import numpy as np
import matplotlib.pyplot as plt
import click
import glob
import os
import time
import json

logger = setup_logging("Feature-Extraction", "ft_extract")
var_envs = get_var_envs()
data_path = os.path.join(var_envs['root'], f"outputs/data/")
plots_path = os.path.join(var_envs['root'], f"outputs/plots/")


def load_data():
    logger.info("========== Loading Data... ==========")
    list_of_files = glob.glob(data_path + "data_clean_*")
    latest_file = max(list_of_files, key=os.path.getctime)
    data = pd.read_csv(latest_file)
    logger.info(f"Features: {list(data.columns)}")
    logger.info(f"Raw data shape: {data.shape}")
    return data


def category_metrics(data, target):
    logger.info(f"Target variable: {target}")
    l_cat = list(set(data[target]))
    logger.info(f"Target unique values: {l_cat}")
    y_cat_num = [(1 - l_cat.index(data.iloc[i][target])) for i in range(len(data))]
    return l_cat, y_cat_num


def get_features(data, feature: str, model):
    logger.info("========== Initializing Model ... ==========")
    logger.info(f"Feature variable: {feature}")
    features = model.fit_transform(data[feature])
    logger.info("Data fitting ...")
    return features


def ari_fct(data,  feature: str, model: object, perplexity, learning_rate, l_cat, y_cat_num):
    # time1 = time.time()
    features = get_features(data, feature, model)
    num_labels = len(l_cat)
    tsne = manifold.TSNE(n_components=2, perplexity=perplexity, n_iter=2000,
                         init='random', learning_rate=learning_rate, random_state=42)
    x_tsne = tsne.fit_transform(features)

    # Détermination des clusters à partir des données après Tsne
    cls = cluster.KMeans(n_clusters=num_labels, n_init=100, random_state=42)
    cls.fit(x_tsne)
    ari_score = np.round(metrics.adjusted_rand_score(y_cat_num, cls.labels_), 4)
    # time2 = np.round(time.time() - time1, 0)
    # logger.info(f"ARI: {ari_score}  --  Time : {time2}")
    # time2 = np.round(time.time() - time1, 0)
    # print("ari_score : ", ari_score, "time : ", time2)
    return ari_score, x_tsne, cls.labels_


def tsne_visu_fct(x_tsne, labels, l_cat, y_cat_num) -> None:
    logger.info("========== Data Visualization  ==========")

    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(121)
    scatter = ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_cat_num, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=l_cat, loc="best", title="Category")
    plt.title('Description per category')

    ax = fig.add_subplot(122)
    scatter = ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=labels, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=set(labels), loc="best", title="Clusters")
    plt.title('Description par clusters')
    plt.savefig(plots_path + 'tsne_kmeans_plot.png')
    logger.info("Visuals saved to : outputs/plots/tsne_kmeans_plot.png")
    plt.show()


def test_model(data: object,
               target: str,
               feature: str,
               model: object,
               tuning_params: tuple[list[str]],
               ) -> object:
    """
    @tuning_params : perplexity and learning rate couple lists
    """
    logger.info("========== Model Tuning  ==========")
    logger.info("Perplexity x learning_rate")
    logger.info(f"Model name: {repr(model)}")
    l_cat, y_cat_num = category_metrics(data, target)
    results = {}
    i = 0
    for perplexity in tuning_params[0]:
        for learning_rate in tuning_params[1]:
            time1 = time.time()
            ari_score, x_tsne, labels = ari_fct(data, feature, model, perplexity, learning_rate, l_cat,
                                                y_cat_num)
            time2 = np.round(time.time() - time1, 0)
            logger.info(f"ARI: {ari_score}  --  Time : {time2}  --  Perpexit : {perplexity}  --  Learning_rate : {learning_rate}")
            row = [repr(model), ari_score, perplexity, learning_rate]
            results[i] = row
            i += 1

    # results_df = pd.DataFrame.from_dict(results, columns=["model", "ari_score", "perplexity", "learning_rate"],
    #                                     orient='index')
    # results_df.to_csv(data_path + "results.csv")

    with open(data_path + "results.json", "w") as outfile:
        json.dump(results, outfile)
    tsne_visu_fct(x_tsne, labels, l_cat, y_cat_num)


@click.command
@click.pass_context
def extract_features(ctx: click.Context):
    logger.info("========== Initializing ==========")
    data = load_data()
    test_model(data, 'category', 'text',
               CountVectorizer(),
               ([20, 30, 40, 50], [100, 200, 300]))
