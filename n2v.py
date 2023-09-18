import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from node2vec import Node2Vec as n2v
import geopandas as gpd
import os
import osmnx as ox
from descartes import PolygonPatch
from shapely.geometry import Point, LineString, Polygon
from tqdm import tqdm
from scipy.stats import pearsonr
import datetime
from pyproj import Transformer
import folium
import random
from IPython.display import display
import torch
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import time
import tensorflow as tf
sns.set()

#GRAPH
ox.config(use_cache=True, log_console=True)
G = ox.graph_from_place('Pinerolo, Italy', simplify=True, network_type='all')
G = ox.project_graph(G, to_crs=4326)
G = G.subgraph(max(nx.strongly_connected_components(G), key=len))
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
print("Numero di Nodi grafo:", G.number_of_nodes())
print("Numero di Archi grafo:", G.number_of_edges())

#DEGREE DISTRIBUTION
plt.clf()
plt.hist(list(dict(G.degree()).values()))
plt.title('Degree Distribution', fontsize=15, pad=15)
plt.ylabel('Numero di nodi', fontsize=11)
plt.xlabel('Numero di connessioni', fontsize=11)
plt.tight_layout()
plt.show()

#N2V
#   compute node2vec
g_emb = n2v(G, dimensions=100)

#   explore node2vec
WINDOW = 1
MIN_COUNT = 1
BATCH_WORDS = 4

mdl = g_emb.fit(
    window=WINDOW,
    min_count=MIN_COUNT,
    batch_words=BATCH_WORDS
)

#SET UP THE TRAINING SET
num_training=1000
rand_nodes = random.sample(list(G.nodes), num_training*2)
orig_nodes = rand_nodes[:num_training]
dest_nodes = rand_nodes[num_training:]
nm_list=[]
x=[] #input
y=[] #output
for i in range(0, num_training):
    orig_node=orig_nodes[i]
    dest_node=dest_nodes[i]
    distance=nx.shortest_path_length(G, orig_node, dest_node, weight='length')
    vect_avg=(mdl.wv.vectors[mdl.wv.key_to_index[str(orig_node)]]+mdl.wv.vectors[mdl.wv.key_to_index[str(dest_node)]])/2
    nm_list.append(((orig_node, dest_node), vect_avg, distance))
    x.append(vect_avg)
    y.append(distance)

seed_random=9000
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=np.random.seed(seed_random), shuffle=True)
print(x_train[0])

#TRAIN THE NETWORK
print(tf.config.list_physical_devices('GPU'))
NN_model = Sequential()
NN_model.add(Dense(100, kernel_initializer='normal', input_dim = len(x_train[1]), activation='linear')) #input layer
NN_model.add(Dense(50, kernel_initializer='normal', activation='linear')) #hidden layer
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear')) #output layer
#   Compile the network
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error']) #configurazione modello
NN_model.summary()
x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)
NN_model.fit(x_train, y_train, epochs=10, batch_size=8, validation_split = 0.2)
predictions = NN_model.predict(x_test)
#RESULTS
print('PREDICTIONS: (', len(predictions), ')')
for i in range(0,len(predictions)):
    print(predictions[i], ' ==> ', y_test[i], ' || ', abs(predictions[i]-y_test[i]))

MAE = mean_absolute_error(y_test , predictions)
print('MAE = ', MAE)
