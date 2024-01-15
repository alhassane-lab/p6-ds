from gensim.models import Word2Vec
from notebooks import fcp_NLP, fct_clustering
import os
import glob
import pandas as pd
from envconf import get_var_envs

# Extraction de features :


ROOT = get_var_envs()['root']


def load_data():
    data_path = os.path.join(ROOT, f"outputs/data/")
    list_of_files = glob.glob(data_path + "data_clean_*")
    latest_file = max(list_of_files, key=os.path.getctime)
    data = pd.read_csv(latest_file)

    return data


data = load_data()
# Création du modèle :
model = Word2Vec(
    sentences=data['stem_desc'],
    vector_size=300,
    window=4,
    min_count=3,
    sg=0,
)


df = fcp_NLP.mean_embending_w2v_df(
    descriptions=data['stem_desc'].tolist(),
    word2vec_model=model,
)

X = df.values

print(f"Dimension du DataFrame de vectorisation : {df.shape}")

kmeans_params, tsne_perplexity, labels, dict_result = \
    fct_clustering.best_clustering(
        features=X,
        y_true=data['stem_desc'],
        tsne_perplexity_range=[5, 10, 20, 30, 40, 50],
        kmeans_clusters=data['stem_desc'].nunique(),
        ls_ninit=[10, 20, 30, 50, 70, 80, 100],
        ls_maxiter=[100, 200, 300, 400, 500],
        title='w2v - CBOW - Descriptions tokenisées avec les stop_words',
        )




#
# if __name__ == "__main__":
#