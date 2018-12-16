import pandas as pd
import gensim
from helpers import sentence_to_vec
from helpers import normalization
from sklearn.cluster import KMeans
import numpy as np

print("loading model")
Vec = gensim.models.KeyedVectors.load_word2vec_format('.\GoogleNews-vectors-negative300.bin', binary=True)
print("loaded model!")
model = None
try:
    model = Vec.wv
except Exception:
    model = Vec

data = pd.read_csv("data.csv", encoding='latin-1')

def preprocessing(data):
    # print(data.head())
    # print(data.shape)
    # print(data.dtypes)
    # print(data.isnull().sum())
    
    data = data[pd.notnull(data['1'])]
    data = data[pd.notnull(data['0'])]
    
    data.columns = ['Text', 'Class']    
    data['Class'] = data['Class'].astype('int')
    data['Text'] = data['Text'].astype('str')
    return data
    
data_new = preprocessing(data)
text = data_new['Text']

features = []
for sent in text:
    sent = normalization(sent)
    features.append(sentence_to_vec(sent, model))

n_clusters = 8
km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1)
km.fit(features)
print([list(km.labels_).count(k) for k in range(n_clusters)])
for i in range(n_clusters):
    indices = [k for k in range(len(km.labels_)) if km.labels_[k] == i]
    cluster_text = text.iloc[indices]
    cluster_text.index = np.arange(len(indices))
    cluster_text.to_csv("clusters/cluster_" + str(i + 1) + ".csv")