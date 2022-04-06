import pandas as pd
import numpy as np
import matplotlib as mpl

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import manifold, datasets
from sklearn.metrics import silhouette_score
from sklearn.neighbors import DistanceMetric

if __name__ == "__main__":

    data_normalised = pd.read_csv(r'/home/tzloop/Desktop/Pointctl/data/EVS/wine-src.csv', sep=";")

    m = manifold.TSNE(n_components=2, init='pca', random_state=111)
    tsne_features = m.fit_transform(data_normalised)
    pd.DataFrame(tsne_features).to_csv(r'/home/tzloop/Desktop/Pointctl/data/EVS/wine-2D.csv', sep=";", index=False)

    n = manifold.TSNE(n_components=3, init='pca', random_state=111)
    tsne_features = n.fit_transform(data_normalised)
    pd.DataFrame(tsne_features).to_csv(r'/home/tzloop/Desktop/Pointctl/data/EVS/wine-3D.csv', sep=";", index=False)