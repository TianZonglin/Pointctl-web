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
from sklearn import preprocessing
if __name__ == "__main__":

    #data = pd.read_csv('https://cloud.hollander.online/s/Z8D9HbWJWJMP2Hf/download')
    #data_filtered = data.filter(['v6', 'v9', 'v36', 'v51', 'v52', 'v54', 'v55', 'v56', 'v57', 'v58', 'v59', 'v60', 'v61', 'v62', 'v63', 'v64', 'v93', 'v115', 'v134', 'v196'])
    #data_filtered.to_csv(r'/home/tzloop/Desktop/Pointctl/data/EVS/filtered.csv', index=False) 
    data_filtered = pd.read_csv(r'/home/tzloop/Desktop/Pointctl/data/EVS/filtered.csv')
    data_filtered = data_filtered.filter(['v6', 'v9', 'v36', 'v52', 'v54', 'v55', 'v56', 'v57', 'v58', 'v59', 'v60', 'v61', 'v62', 'v63', 'v64', 'v93', 'v115', 'v134', 'v196'])



    data_filtered["v6"].replace({'very important': 1, 'quite important': 2, 'not important': 3, 'not at all important': 4, 'dont know':8, 'no answer':9}, inplace=True)
    data_filtered["v36"].replace({'trust completely': 1, 'trust somewhat': 2, 'do not trust very much': 3, 'do not trust at all': 4, 'dont know':8, 'no answer':9}, inplace=True)
    data_filtered["v54"].replace({'more than once week': 1, 'once a week': 2, "once a month": 3, "only on specific holy days": 4, "once a year": 5, "less often": 6, "never, practically never": 7, 'dont know':8, 'no answer':9, 'multiple answers Mail': 101}, inplace=True)
    data_filtered["v55"].replace({'more than once week': 1, 'once a week': 2, "once a month": 3, "only on specific holy days": 4, "once a year": 5, "less often": 6, "never, practically never": 7, 'dont know':8, 'no answer':9, 'multiple answers Mail': 101}, inplace=True)
    data_filtered["v56"].replace({'a religious person': 1, 'not a religious person': 2, "a convinced atheist": 3, 'dont know':8, 'no answer':9}, inplace=True)
    data_filtered["v63"].replace({'not at all important': 1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, "very important": 10, 'dont know':88, 'no answer':99}, inplace=True)
    data_filtered["v9"].replace({'mentioned': 1, 'not mentioned': 2, 'dont know':8, 'no answer':9}, inplace=True)
    #data_filtered["v51"].replace({'yes': 1, 'no': 2, 'dont know':8, 'no answer':9}, inplace=True)
    data_filtered["v52"].replace({'Muslim': 1, 'Orthodox': 2, 'not applicable': 77, 'Roman catholic': 3,
        'Protestant': 4, 'Other': 5, 'no answer': 99, 'dont know': 88, 'Jew': 6, 'Buddhist': 7,
        'Free church/Non-conformist/Evangelical': 8, 'Hindu': 9}, inplace=True)
    data_filtered["v57"].replace({'yes': 1, 'no': 2, 'dont know':8, 'no answer':9}, inplace=True)
    data_filtered["v58"].replace({'yes': 1, 'no': 2, 'dont know':8, 'no answer':9}, inplace=True)
    data_filtered["v59"].replace({'yes': 1, 'no': 2, 'dont know':8, 'no answer':9}, inplace=True)
    data_filtered["v60"].replace({'yes': 1, 'no': 2, 'dont know':8, 'no answer':9}, inplace=True)
    data_filtered["v61"].replace({'yes': 1, 'no': 2, 'dont know':8, 'no answer':9}, inplace=True)
    data_filtered["v62"].replace({'personal God': 1, 'spirit or life force': 2, "don't know what to think": 3, "no spirit, God or life force": 4, 'dont know':8, 'no answer':9}, inplace=True)
    data_filtered["v93"].replace({'not mentioned': 2, 'mentioned': 1, 'no answer': 9, 'dont know': 8}, inplace=True)
    data_filtered["v64"].replace({'every day': 1, 'more than once week': 2, 'once a week': 3,  'at least once a month': 4, 'several times a year': 5,  'less often': 6, 'never': 7, 'no answer': 9, 'dont know': 8, 'multiple answers Mail': 101, 'other missing': 100}, inplace=True)
    data_filtered["v115"].replace({'a great deal': 1, 'quite a lot': 2, 'not very much': 3, 'none at all': 4, 'dont know':8, 'no answer':9}, inplace=True)
    data_filtered["v134"].replace({'not at all an essential characteristic of democracy': 1, 'dont know': 88, 'no answer': 99, 'an essential characteristic of democracy': 10 ,'it is against democracy [DO NOT READ OUT]': 0, 'item not included': 100, 'multiple answers Mail': 101}, inplace=True)
    data_filtered["v196"].replace({'very important': 1, 'quite important': 2, 'not important': 3, 'not at all important': 4, 'dont know':8, 'no answer':9}, inplace=True)


    data_cleaning = data_filtered
 

    data_cleaning = data_cleaning[data_cleaning.v6 != 8]
    data_cleaning = data_cleaning[data_cleaning.v6 != 9]

    data_cleaning = data_cleaning[data_cleaning.v36 != 8]
    data_cleaning = data_cleaning[data_cleaning.v36 != 9]

    data_cleaning = data_cleaning[data_cleaning.v54 != 8]
    data_cleaning = data_cleaning[data_cleaning.v54 != 9]
    data_cleaning = data_cleaning[data_cleaning.v54 != 101]

    data_cleaning = data_cleaning[data_cleaning.v55 != 8]
    data_cleaning = data_cleaning[data_cleaning.v55 != 9]
    data_cleaning = data_cleaning[data_cleaning.v55 != 101]

    data_cleaning = data_cleaning[data_cleaning.v56 != 8]
    data_cleaning = data_cleaning[data_cleaning.v56 != 9]

    data_cleaning = data_cleaning[data_cleaning.v63 != 88]
    data_cleaning = data_cleaning[data_cleaning.v63 != 99]

    data_cleaning = data_cleaning[data_cleaning.v9 != 8]
    data_cleaning = data_cleaning[data_cleaning.v9 != 9]

    #data_cleaning = data_cleaning[data_cleaning.v51 != 8]
    #data_cleaning = data_cleaning[data_cleaning.v51 != 9]

    data_cleaning = data_cleaning[data_cleaning.v52 != 77]
    data_cleaning = data_cleaning[data_cleaning.v52 != 88]
    data_cleaning = data_cleaning[data_cleaning.v52 != 99]

    data_cleaning = data_cleaning[data_cleaning.v57 != 8]
    data_cleaning = data_cleaning[data_cleaning.v57 != 9]
    data_cleaning = data_cleaning[data_cleaning.v58 != 8]
    data_cleaning = data_cleaning[data_cleaning.v58 != 9]
    data_cleaning = data_cleaning[data_cleaning.v59 != 8]
    data_cleaning = data_cleaning[data_cleaning.v59 != 9]
    data_cleaning = data_cleaning[data_cleaning.v60 != 8]
    data_cleaning = data_cleaning[data_cleaning.v60 != 9]
    data_cleaning = data_cleaning[data_cleaning.v61 != 8]
    data_cleaning = data_cleaning[data_cleaning.v61 != 9]

    data_cleaning = data_cleaning[data_cleaning.v62 != 8]
    data_cleaning = data_cleaning[data_cleaning.v62 != 9]

    data_cleaning = data_cleaning[data_cleaning.v93 != 8]
    data_cleaning = data_cleaning[data_cleaning.v93 != 9]

    data_cleaning = data_cleaning[data_cleaning.v64 != 8]
    data_cleaning = data_cleaning[data_cleaning.v64 != 9]
    data_cleaning = data_cleaning[data_cleaning.v64 != 100]
    data_cleaning = data_cleaning[data_cleaning.v64 != 101]


    data_cleaning = data_cleaning[data_cleaning.v115 != 8]
    data_cleaning = data_cleaning[data_cleaning.v115 != 9]

    data_cleaning = data_cleaning[data_cleaning.v134 != 88]
    data_cleaning = data_cleaning[data_cleaning.v134 != 99]
    data_cleaning = data_cleaning[data_cleaning.v134 != 100]
    data_cleaning = data_cleaning[data_cleaning.v134 != 101]

    data_cleaning = data_cleaning[data_cleaning.v196 != 8]
    data_cleaning = data_cleaning[data_cleaning.v196 != 9]


 
    data_normalised = data_cleaning




    #normalise_subset = [['v6', 'v36', 'v52', 'v54', 'v55', 'v56', 'v62', 'v63', 'v64', 'v115', 'v134', 'v196']]
    #normalise_subset = data_normalised
    #data_normalised = data_normalised.filter(['v6', 'v9', 'v36', 'v52', 'v54', 'v55', 'v56', 'v57', 'v58', 'v59', 'v60', 'v61', 'v62', 'v63', 'v64', 'v93', 'v115', 'v134', 'v196'])
    

    x = data_normalised.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data_normalised = pd.DataFrame(x_scaled)

    data_normalised.columns = data_filtered.columns

    data_normalised["v9"].replace( {0: 0.7071,1: 0}, inplace=True)
    #data_normalised["v51"].replace({0: 0.7071,1: 0}, inplace=True)
    data_normalised["v57"].replace({0: 0.7071,1: 0}, inplace=True)
    data_normalised["v58"].replace({0: 0.7071,1: 0}, inplace=True)
    data_normalised["v59"].replace({0: 0.7071,1: 0}, inplace=True)
    data_normalised["v60"].replace({0: 0.7071,1: 0}, inplace=True)
    data_normalised["v61"].replace({0: 0.7071,1: 0}, inplace=True)
    data_normalised["v93"].replace({0: 0.7071,1: 0}, inplace=True)

    #normalise_subset = (normalise_subset-normalise_subset.min())/(normalise_subset.max()-normalise_subset.min())

    #data_normalised.update(normalise_subset)

    filename = "evs-without-v51"

    data_normalised.to_csv('/home/tzloop/Desktop/Pointctl/data/EVS/'+filename+'-src.csv', index=False, sep=";")
 





    #m = TSNE(n_components=2, init='pca', learning_rate=1500, perplexity=125, random_state=234, verbose=1)
    m = TSNE(n_components=2, init='random',learning_rate=1500, perplexity=125, random_state=234, verbose=1)
    tsne_features = m.fit_transform(data_normalised)
    pd.DataFrame(tsne_features).to_csv('/home/tzloop/Desktop/Pointctl/data/EVS/'+filename+'-2d.csv', sep=";", index=False)

    #n = TSNE(n_components=3, init='random', learning_rate=1500, perplexity=125, random_state=234, verbose=1)
    n = TSNE(n_components=3, init='random', learning_rate=1500, perplexity=125, random_state=234, verbose=1)
    tsne_features = n.fit_transform(data_normalised)
    pd.DataFrame(tsne_features).to_csv('/home/tzloop/Desktop/Pointctl/data/EVS/'+filename+'-3d.csv', sep=";", index=False)
    