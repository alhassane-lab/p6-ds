
from sys import displayhook
import numpy as np
import pandas as pd
from itertools import permutations

## Graphique
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# PIL
from PIL import Image

## Scikit-Learn
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.metrics import completeness_score, homogeneity_score, v_measure_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


###########################################################################################################################
## CLASSIFICATION NON SUPERVISEE

def proj_tsne(features, tsne_perplexity):
    """
    Calcul la projection TSNE en 2 dimenssions des features en fonction de la perplxité donnée.

    Args:
        features (np.array): Features à réduire dimensionnellement
        tsne_perplexity (int): Perplexité à utiliser pour la réduction TSNE

    Returns:
        np.array: Tableau des variables de réduction TSNE en 2D.
    """

    # 1. Initialisation d'un tsne
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, init='random', random_state=42)

    # 2. Création de la projection tsne :
    X_tsne = tsne.fit_transform(features)

    return X_tsne


def test_perplexity_tsne(features, y_true, ls_perplexity):
    """
    Affiche les différentes représentation graphique des projections en fonction de différentes valeurs de perplexité pour TSNE.


    Args:
        features (np.array): Tableau numpy contenant les extractions de features
        y_true (pd.Series): Serie pandas de la cible (catégories réelles des produits)
        ls_perplexity (list): liste des différentes perplexité à utiliser
    """

    # 1. Création de la figure :
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))

    for i, perplexity in enumerate(ls_perplexity):
        # 2. Création de la projection tsne :
        tsne = TSNE(n_components=2, perplexity=perplexity, init="random", random_state=42)
        X_tsne = tsne.fit_transform(features)

        # 3. Calcul de la ligne et de la colonne du graphique :
        row = i // 2
        col = i % 2

        # 4. Création du graphique :
        ax = axes[row, col]

        sns.scatterplot(x=X_tsne[:, 0],
                        y=X_tsne[:, 1],
                        hue=y_true,
                        palette='viridis',
                        ax=ax,
                        legend=False,
                        )
        ax.set_title(f"Projection tsne - perplexité = {perplexity}")

    plt.tight_layout()
    plt.show()


def calcul_metrics_kmeans(X_tsne, kmeans_clusters, ls_ninit, ls_maxiter):
    """
    Test de différents hyperparamètres de KMeans pour déterminer la meilleures combinaison.

    Args:
        X_tsne (np.array): Tableau numpy contenant les réductions des features en 2D
        kmeans_clusters (int): Nombre de clusters pour Kmeans
        ls_ninit (list): Liste des n_init à tester pour Kmeans
        ls_maxiter (list): Liste des max_iter à tester pour KMeans

    Returns:
        DateFrame: Dataframe des résultats pour les différentes combinaison d'hyperparamètres.
    """

    # 1. Initialisation des listes de stockage :
    silhouette = []
    dispersion = []
    davies_bouldin = []

    result_ninit = []
    result_maxiter = []

    # 2. Recherche des hyperparamètres :
    for ninit in ls_ninit:

        for maxiter in ls_maxiter:
            # 2.1 Initialisation de l'algorithme :
            kmeans = KMeans(n_clusters=kmeans_clusters,
                            n_init=ninit,
                            max_iter=maxiter,
                            random_state=42)

            # 2.2 Entraînement de l'algorithme :
            kmeans.fit(X_tsne)

            # 2.3 Prédictions :
            preds = kmeans.predict(X_tsne)

            # 3. Calcul du score de coefficient de silhouette :
            silh = silhouette_score(X_tsne, preds)

            # 4. Calcul la dispersion :
            disp = kmeans.inertia_

            # 5. Calcul de l'indice davies-bouldin :
            db = davies_bouldin_score(X_tsne, preds)

            # 6. Enregitrement des résultats :
            silhouette.append(silh)
            dispersion.append(disp)
            davies_bouldin.append(db)

            result_ninit.append(ninit)
            result_maxiter.append(maxiter)

    # 7. Création du DataFrame :
    dataframe_metrique = pd.DataFrame({
        'n_init': result_ninit,
        'max_iter': result_maxiter,
        'coef_silh': silhouette,
        'dispersion': dispersion,
        'davies_bouldin': davies_bouldin,
    })

    dataframe_metrique = dataframe_metrique.sort_values(
        by=['coef_silh', 'dispersion', 'davies_bouldin'],
        ascending=[False, True, True],
    )
    return dataframe_metrique


def calcul_ARI(X_tsne, y_true, kmeans_params):
    """
    Calcul l'ARI entre les labels (créés par KMeans avec les hyper-paramètres donnés et la projection TSNE) et les catégories réelles des produits

    Args:
        X_tsne (np.array): Tableau numpy contenant les réductions des features en 2D
        y_true (pd.Series): Serie pandas de la cible (catégories réelles des produits)
        kmeans_params (dict): dictionnaire des paramètres pour KMeans

    Returns:
        (int, pd.Serie): tuple du score ARI et des labels créés par KMeans.
    """

    # 1. Création de KMeans :
    clustering_model = KMeans(**kmeans_params)
    clustering_model.fit(X_tsne)

    # 2. Calcul de l'ARI :
    ARI = np.round(adjusted_rand_score(y_true, clustering_model.labels_), 4)

    return ARI, clustering_model.labels_


def visu_clustering(X_tsne, y_true, y_label, title):
    """
   Trace les projection dans l'espace réduit en 2D par TSNE des répartitions des produits en fonction
   des vrais catégories et des clusters crées par KMeans

   Args:
        X_tsne (np.array): Tableau numpy contenant les réductions des features en 2D
        y_true (pd.Series): Serie pandas de la cible (catégories réelles des produits)
        y_label (pd.Series): Serie pandas des clusters créés par KMeans
        title (str): Titre à afficher avant le tracé des graphiques
   """

    print('--' * 50)
    print("Comparaison des projections du clustering vs catégories réelles\n->\n" + title)
    print('--' * 50)

    fig, axes = plt.subplots(2, 1, figsize=(8, 12))

    sns.scatterplot(x=X_tsne[:, 0],
                    y=X_tsne[:, 1],
                    hue=y_true,
                    palette='viridis',
                    ax=axes[0],
                    )
    axes[0].set_title('Représentation des catégories de produits par valeurs réelles')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Categorie')

    sns.scatterplot(x=X_tsne[:, 0],
                    y=X_tsne[:, 1],
                    hue=y_label,
                    palette='viridis',
                    ax=axes[1],
                    )
    axes[1].set_title('Représentation des catégories de produits par clusters')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Clusters')

    plt.show()


def best_clustering(features, y_true, tsne_perplexity_range, kmeans_clusters, ls_ninit, ls_maxiter, title):
    """
    Réduction dimensionnelles avec TSNE en 2D des features.
    Trace les graphique de comparaison des perplexité de TSNE.
    Calcul les meilleurs paramètres de KMeans en fonction des perplexités se basant sur l'ARI.
    Calcul les indices d'évaluations du clustering pour les meilleurs paramètres de Kmeans.
    Affiche les résultats et les tracés de comparaison de la répartition des produits dans les vraie catégories
    versus les clusters créés par Kmeans.

    Args:
        features (np.array): Features à réduire dimensionnellement
        y_true (pd.Series): Serie pandas de la cible (catégories réelles des produits)
        tsne_perplexity_range (list): liste des différentes perplexité à utiliser
        kmeans_clusters (int): Nombre de clusters pour Kmeans
        ls_ninit (list): Liste des n_init à tester pour Kmeans
        ls_maxiter (list): Liste des max_iter à tester pour KMeans
        title (str): Titre à passer pour le tracé des graphiques

    Returns:
        (dict, int, pd.Series, dict): tupple contenant :
        - le dictionnaire des meilleurs paramètres de KMeans
        - La meilleure perplexité pour la réduction de dimensions avec TSNE
        - La série pandas des clusters créés
        - le dictionnaire des résultats d'évaluation du clusteting (ARI, Homogénéité, Complétude, V-measure)
    """
    # Ignorer les avertissements
    warnings.filterwarnings("ignore")

    # 1. Affichage des projections TSNE en fonction des perplexité :
    print('--' * 50)
    print(f"Projections en 2D avec réduction TSNE en fonction de la perplexité")
    test_perplexity_tsne(
        features=features,
        y_true=y_true,
        ls_perplexity=tsne_perplexity_range,
    )

    # 2. Création des données de stockage :
    best_ARI = -1
    kmeans_params = {
        'n_clusters': y_true.nunique(),
        'random_state': 42,
    }

    for perplexity in tsne_perplexity_range:

        # 3. Calcul des projections de features :
        X_tsne = proj_tsne(
            features=features,
            tsne_perplexity=perplexity,
        )

        # 4. Calcul des résultats pour les combinaisons d'hyperparamètres :
        df = calcul_metrics_kmeans(
            X_tsne=X_tsne,
            kmeans_clusters=kmeans_clusters,
            ls_ninit=ls_ninit,
            ls_maxiter=ls_maxiter,
        )

        kmeans_params['n_init'] = df.loc[0, 'n_init']
        kmeans_params['max_iter'] = df.loc[0, 'max_iter']

        # 5. Calcul du score ARI :
        ARI, y_labels = calcul_ARI(
            X_tsne=X_tsne,
            y_true=y_true,
            kmeans_params=kmeans_params,
        )

        # 6. Comparaison des ARI :
        if ARI > best_ARI:
            best_ARI = ARI
            best_kmeans_params = kmeans_params
            best_perplexity = perplexity
            best_kmeans_results = df.iloc[0, 2:].to_frame().T
            best_X_tsne = X_tsne
            best_y_labels = y_labels
            best_homogeneity = homogeneity_score(y_true, best_y_labels)
            best_completeness = completeness_score(y_true, best_y_labels)
            best_v_measure = v_measure_score(y_true, best_y_labels)

    # 7. Affichage des résultats :

    print('--' * 50)
    print(f"Meilleure perplexité de TSNE : {best_perplexity}")
    print('--' * 50)
    print(f"Meilleurs paramètres de KMeans : {best_kmeans_params}")
    displayhook(best_kmeans_results)
    print('--' * 50)
    print(f"Score ARI du clustering : {best_ARI}")
    print(f"Homogénéité : {best_homogeneity}")
    print(f"Complétude : {best_completeness}")
    print(f"V-measure : {best_v_measure}")

    visu_clustering(
        X_tsne=best_X_tsne,
        y_true=y_true,
        y_label=best_y_labels,
        title=title,
    )
    # Réactiver les avertissements si nécessaire
    warnings.filterwarnings("default")

    # 8. Renvoie des résultats :

    dict_result = {
        'ARI': best_ARI,
        'Homogénéité': best_homogeneity,
        'Complétude': best_completeness,
        'V-measure': best_v_measure,
    }

    return best_kmeans_params, best_perplexity, best_y_labels, dict_result


def calcul_inertia(features, k_range):
    """
    Calcul les inerties du clusturing de KMeans en fonction de différents nombres de clusters

    Args:
        features (list or np.array):  features pour l'entrainement de KMeans
        k_range (list): liste des différents nombre de clusters à tester

    Returns:
        list: liste des inerties pour chaque entrainement de KMeans
    """

    # 1. Création de la liste de stockage :
    inertia = []

    # 2. Test des valeurs :
    for k in tqdm(k_range, desc='Testing K values'):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        inertia.append(kmeans.inertia_)

    return inertia


def graph_elbow_method(inertia, k_range):
    """
    Trace le graphique de la méthode du coude

    Args:
        inertia (list): liste des interties calculer de KMeans
        k_range (list): liste des nombres de clusters pour l'entrainement de KMeans
    """

    plt.figure(figsize=(12, 8))

    plt.plot(k_range, inertia, marker='o')
    plt.xlabel('Nombre de clusters (K)')
    plt.ylabel('Inertie')
    plt.title('Méthode du Coude')

    plt.tight_layout()
    plt.show()


###########################################################################################################################
## 5. RESULTATS


def classement_resultats(dico):
    """
    Construction d'un DataFrame de classement en fonction de l'ARI sur l'ensemble des modèles et Tracé des résultats.

    Args:
        dico (dict): Dictionnaire où sont stockés l'ensemble des résultats de l'étude.
    """

    # 1. Création des listes pour stocker les données :
    modeles = []
    ari_scores = []
    homogeneite_scores = []
    completude_scores = []
    v_measure_scores = []

    # 2. Récupération des résultats :
    for modele, valeurs in dico.items():
        modeles.append(modele)
        dict_result = valeurs['clustering_result']
        ari_scores.append(dict_result['ARI'])
        homogeneite_scores.append(dict_result['Homogénéité'])
        completude_scores.append(dict_result['Complétude'])
        v_measure_scores.append(dict_result['V-measure'])

    # 3. Création du DataFrame :
    df = pd.DataFrame({
        'Modèle': modeles,
        'ARI': ari_scores,
        'Homogénéité': homogeneite_scores,
        'Complétude': completude_scores,
        'V-measure': v_measure_scores
    })

    df.sort_values('ARI', ascending=False, inplace=True)
    df.set_index('Modèle', inplace=True)
    displayhook(df)

    # 4. Affichage du graphique :
    df.sort_values('ARI', ascending=True, inplace=True)
    name = df.index
    position = 2 * np.arange(len(df))
    height = 0.25

    plt.figure(figsize=(12, 12))

    plt.barh(position + 1.5 * height, df['ARI'], height=height, label='ARI', color='b')
    plt.barh(position + 0.5 * height, df['Homogénéité'], height=height, label='Homogénétié', color='g')
    plt.barh(position - 0.5 * height, df['Complétude'], height=height, label='Complétude', color='r')
    plt.barh(position - 1.5 * height, df['V-measure'], height=height, label='V-measure', color='y')

    plt.yticks(position, name)
    plt.legend(loc='best')
    plt.title('Comparaison des scores des clusterings')
    plt.show()


def generate_cluster_dataframe(col_label, title):
    """
    Renvoie un tableau du nombre d'occurences par clusters et du % associé.
    Trace le graphique de répartition par clusters

    Args:
        col_label (pd.Series): Série pandas des clusters
        title (str): Nom du modèle d'extraction de features à ajouter au titre du graphique.
    """

    # 1. Compte du nombre d'occurrence de chaque cluster :
    cluster_counts = col_label.value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Nombre']

    # 2. Calcul du pourcentage :
    cluster_counts['%'] = (cluster_counts['Nombre'] / len(col_label)) * 100

    # 3. Trie du DataFrame :
    cluster_counts = cluster_counts.sort_values(by='Cluster', ascending=True)

    # 4. Affichage du DtaFrame :
    displayhook(cluster_counts)

    # 5. Affichage du graphique :

    plt.figure()
    plt.bar(x=cluster_counts['Cluster'], height=cluster_counts['Nombre'])
    plt.title(f"Répartition par clusters -\n{title}")
    plt.xlabel('Clusters')
    plt.ylabel('Nombre')
    plt.show()


def qualite_categorisation(df, dico_trad, title):
    """
    Trace la matrice de confusion
    Renvoie le tableau de rapport des métriques

    Args:
        df (DataFrame): DataFrame de vecotrisation avec les colonnes de catégories réelles et les labels
        dico_trad (dict): Dictionnaire du lien entre les catégories et les labels
        title (str): Titre de la matrice de confusion.
    """

    # 1. Définition des y_true et y_pred :
    y_pred = df['Category_predict'].copy().map(dico_trad)
    y_true = df['Category_true'].copy()

    # 2. Calcul de la martrice de confusion :
    matrice_confusion = confusion_matrix(y_true, y_pred)

    # 3. Affichage de la matrice :
    plt.figure(figsize=(6, 4))

    sns.heatmap(matrice_confusion,
                annot=True,
                fmt='d',
                cbar=False,
                cmap='Blues',
                xticklabels=list(dico_trad.keys()), yticklabels=list(dico_trad.values()),
                )

    plt.xlabel('Prédictions')
    plt.ylabel('Vraies Catégories')
    plt.title(title)

    plt.show()

    # 4. Affichage du rapport de classification  :
    report = classification_report(y_true, y_pred)
    print(report)


def dataframe_comp_result(df_init, df_categ, dico_trad, chemin, type='texte'):
    """
    Construit et affiche un DataFrame avec des produits ayant une mauvaise classification, ainsi qu'une liste de descriptions, d'images ou les deux pour ces produits.

    Args:
        df_init (pd.DataFrame): dataframe initial avec les noms de produits, les images, etc.
        df_categ (pd.DataFrame): dataframe de vectorisation d'un modèle.
        dico_trad (dict): dictionnaire de traduction pour renommer les clusters.
        type (str): Le type d'affichage souhaité ('texte', 'image', 'texte+image'). Par défaut, 'texte'.
        chemin (str): Le chemin d'accès au dossier image.
    """
    df_categ = df_categ.copy()

    # 1. Création de la traduction des clusters :
    df_categ['Category_predict_trans'] = df_categ['Category_predict'].map(dico_trad)

    # 2. Création du DataFrame de résultats :
    df_result = pd.DataFrame(columns=['product_name', 'description', 'image', 'category_true', 'category_predict'])

    # 3. Boucle sur les categéories :
    for categ in df_categ['Category_true'].unique():

        # 3.1 Création de masques :
        mask_1 = df_categ['Category_true'] == categ
        mask_2 = df_categ['Category_predict_trans'] != categ

        # 3.2 Filtre de df_categ :
        df_filtre = df_categ.loc[mask_1 & mask_2]

        # 3.3 Récupération des index de 3 valeurs au hasard :
        if len(df_filtre) >= 3:
            index = df_filtre.sample(3).index
        else:
            index = df_filtre.index

            # 3.4 Récupération des valeurs :
        lignes = pd.DataFrame({
            'product_name': df_init.loc[index, 'product_name'].values.tolist(),
            'description': df_init.loc[index, 'description'].values.tolist(),
            'image': df_init.loc[index, 'image'].values.tolist(),
            'category_true': df_categ.loc[index, 'Category_true'].values.tolist(),
            'category_predict': df_categ.loc[index, 'Category_predict_trans'].values.tolist(),
        })

        # 3.5 Ajout des lignes à df_result :
        df_result = pd.concat([df_result, lignes], ignore_index=True)

    displayhook(df_result)

    # 4. Affichage en fonction du type spécifié :

    for i in range(len(df_result)):
        print('--' * 50)
        print(f"Produit {i} :")
        print(f"Nom du produit : {df_result.iloc[i, 0]}")
        print(f"Catégorie réelle : {df_result.iloc[i, 3]}")
        print(f"Catégorie prédite : {df_result.iloc[i, 4]}")

        if type == 'texte':
            print(f"Description {i} :")
            print(df_result.iloc[i, 1], '\n')

        elif type == 'image':

            image_path = chemin + "/" + df_result.iloc[i, 2]

            plt.figure()
            img = Image.open(image_path)
            plt.imshow(img)
            plt.show()

        elif type == 'texte+image':

            print(f"Description {i} :")
            print(df_result.iloc[i, 1], '\n')

            image_path = chemin + "/" + df_result.iloc[i, 2]

            plt.figure()
            img = Image.open(image_path)
            plt.imshow(img)
            plt.show()

        else:
            print("Type non valide. Utilisez 'texte', 'image' ou 'texte+image'.")


def repartition_erreurs(df, dico_trad):
    """
    Trace les graphiques de la répartition du nombre d'erreurs par catégories

    Args:
        df (pd.DataFrame): DataFrame de vectorisation
        dico_trad (dict): dictionnaire de traduction pour renommer les clusters
    """
    df = df.copy()

    # 1. Création de la traduction des clusters :
    df['Category_predict_trans'] = df['Category_predict'].map(dico_trad)

    # 2. Filtres sur df :
    df = df.loc[df['Category_true'] != df['Category_predict_trans']]

    # 3. Graphique du nombre d'errreur :
    plt.figure(figsize=(6, 6))

    sns.countplot(y='Category_true',
                  data=df,
                  orient='h',
                  order=df['Category_true'].value_counts().index,
                  )

    plt.title("Répartition du nombre d'erreurs par catégorie")
    plt.xlabel("Nombre d'erreur")
    plt.ylabel("Catégories")
    plt.show()

    # Répartition des erreurs par catégorie
    plt.figure(figsize=(6, 6))
    sns.countplot(y='Category_true',
                  hue='Category_predict_trans',
                  data=df,
                  orient='h',
                  )

    plt.title("Répartition des mauvais clusters par catégorie")
    plt.xlabel("Nombre d'erreur")
    plt.ylabel("Catégories")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    print(f"Les plus grosses erreurs sont dues au clusturing de {df['Category_predict_trans'].mode()[0]}")


def find_best_cluster_mapping(true_categories, cluster_labels):
    """
    Trouve la meilleure correspondance entre les catégories réelles et les clusters pour maximiser l'accuracy.

    Args:
    true_categories (np.Series): Les catégories réelles sous forme de chaînes de caractères.
    cluster_labels (np.Series): Les labels de clusters générés par KMeans.

    Returns:
    dict: Le dictionnaire de correspondance pour la meilleure combinaison.
    """
    # 1. Création d'un ensemble de catégories uniques et de clusters uniques :
    unique_categories = true_categories.unique()
    unique_clusters = cluster_labels.unique()

    # 2. Génération de l'ensemble des permutations possibles des catégories réelles :
    permutations_list = list(permutations(unique_categories))

    # 3. Initialisation des variables pour stocker la meilleure correspondance et la meilleure accuracy.
    best_accuracy = 0
    best_mapping = None

    # 4. Parcours des permutations et calcul de l'accuracy pour chaque combinaisons :
    for perm in permutations_list:
        category_to_cluster = {cluster: category for cluster, category in zip(unique_clusters, perm)}
        mapped_clusters = pd.Series([category_to_cluster[cluster] for cluster in cluster_labels])
        accuracy = accuracy_score(true_categories, mapped_clusters)

        # 5. Mise à jour de la meilleure correspondance :
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mapping = category_to_cluster

    best_mapping = dict(sorted(best_mapping.items()))
    print(f"Meilleure accuracy : {best_accuracy}")
    print(f"Dictionnaire de traduction :")
    displayhook(best_mapping)

    return best_mapping


def find_best_accuracy(true_categories, cluster_labels):
    """
    Même fonction que find_best_cluster_mapping mais sans printer les résultats.

    Args:
    true_categories (np.Series): Les catégories réelles sous forme de chaînes de caractères.
    cluster_labels (np.Series): Les labels de clusters générés par KMeans.

    Returns:
    float: Le score de la meilleure accuracy.
    """
    # 1. Création d'un ensemble de catégories uniques et de clusters uniques :
    unique_categories = true_categories.unique()
    unique_clusters = cluster_labels.unique()

    # 2. Génération de l'ensemble des permutations possibles des catégories réelles :
    permutations_list = list(permutations(unique_categories))

    # 3. Initialisation des variables pour stocker la meilleure correspondance et la meilleure accuracy.
    best_accuracy = 0
    best_mapping = None

    # 4. Parcours des permutations et calcul de l'accuracy pour chaque combinaisons :
    for perm in permutations_list:
        category_to_cluster = {cluster: category for cluster, category in zip(unique_clusters, perm)}
        mapped_clusters = pd.Series([category_to_cluster[cluster] for cluster in cluster_labels])
        accuracy = accuracy_score(true_categories, mapped_clusters)

        # 5. Mise à jour de la meilleure correspondance :
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mapping = category_to_cluster

    best_mapping = dict(sorted(best_mapping.items()))

    return best_accuracy


def classement_final(dico):
    """
    Compare les résultats de plusieurs modèles de clustering et affiche un classement
    basé sur l'accuracy, l'ARI (Adjusted Rand Index), l'homogénéité, la complétude
    et la V-mesure. Crée également des graphiques pour visualiser les comparaisons (Accuracy)

    Args:
        dico (dict): dictionnaire contenant l'emsemble des résultats
    """

    # 1. Création des listes pour stocker les données :
    modeles = []
    ari_scores = []
    homogeneite_scores = []
    completude_scores = []
    v_measure_scores = []
    accuracy = []
    type_donnee = []

    # 2. Récupération des résultats :
    for modele, valeurs in dico.items():
        # 2.1 Récupération des données
        modeles.append(modele)

        type_donnee.append(valeurs['type_donnees'])

        dict_result = valeurs['clustering_result']

        ari_scores.append(
            round(dict_result['ARI'], 2)
        )

        homogeneite_scores.append(
            round(dict_result['Homogénéité'], 2)
        )

        completude_scores.append(
            round(dict_result['Complétude'], 2)
        )

        v_measure_scores.append(
            round(dict_result['V-measure'], 2)
        )

        # 2.2 Calcul de l'accuracy :
        df = dico[modele]['df_vectorisation'].copy()
        # y_true = df['Category_true'].copy()
        # y_pred = df['Category_predict'].copy()

        best_accuracy = find_best_accuracy(
            true_categories=df['Category_true'],
            cluster_labels=df['Category_predict']
        )

        # df = df.groupby('Category_predict')['Category_true']
        # dico_cluster = df.apply(lambda x: x.mode()[0]).to_dict()

        accuracy.append(round(best_accuracy, 2))

    # 3. Création du DataFrame :
    df = pd.DataFrame({
        'Modèle': modeles,
        'Accuracy': accuracy,
        'ARI': ari_scores,
        'Homogénéité': homogeneite_scores,
        'Complétude': completude_scores,
        'V-measure': v_measure_scores,
        'Type_donnee': type_donnee,
    })

    df = df.sort_values(by=['Accuracy', 'ARI'], ascending=[False, False])
    df = df.set_index('Modèle')
    displayhook(df)

    # Graphique 1 - Précisions :
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    ax = sns.barplot(x=df['Accuracy'],
                     y=df.index,
                     hue=df['Type_donnee'],
                     width=0.4
                     )

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Comparaison des Précisions', fontsize=30)

    for p in ax.patches:
        ax.annotate(f"{p.get_width():.2f}",
                    xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                    xytext=(5, 0),
                    textcoords='offset points',
                    ha='left',
                    va='center',
                    fontsize=16,
                    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=16)
    plt.xlim(0, 1)
    plt.xlabel('Précision', fontsize=16)
    plt.ylabel("Modèle d'extraction", fontsize=16)

    plt.show()

    return df
