from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
import pandas as pd

#cc_distance
cc_dist=[]
with open('cc2.dat','r')as f:
    lines = f.readlines()
for line in lines:
    cc_dist.append(float(line))

#label
data_list=[]
with open('data_list2.dat','r')as f:
    lines = f.readlines()  
for line in lines:
    data_list.append(str(line))

D = squareform(cc_dist)
"""
#MDS
D = squareform(cc_dist)
mds = MDS(n_components=3, dissimilarity="precomputed", random_state=0)
pos = mds.fit_transform(D)

res = pd.DataFrame(pos, columns=['x', 'y','z'])
"""

#DBSCAN
eps=0.6
min_sample=7

db = DBSCAN(eps=eps, min_samples=min_sample)

dbscan = db.fit_predict(D)
labels = db.labels_

#DBSCAN_color
cluster_num = len(set(labels))
color_label=[]
for i in range(cluster_num):
    color_label.append([])
labels_plus= labels + 1

for i, j  in enumerate(labels_plus):
    color_label[j].append(data_list[i])

"""
#dendrogram
Z = linkage(cc_dist, method='ward', metric='euclidean')
dendrogram(Z,labels=data_list)

ax = plt.gca()
xlabels = ax.get_xmajorticklabels()
for label in xlabels:
    if 'apo' in label.get_text():
        label.set_color("red")
    elif 'ben' in label.get_text():
        label.set_color("blue")
    elif 'try' in label.get_text():
        label.set_color("green")
        
plt.savefig('dendro_1.png')
plt.clf()
"""

#dendrogram2
Z = linkage(cc_dist, method='ward', metric='euclidean')
dendrogram(Z,labels=data_list)

ax = plt.gca()
xlabels = ax.get_xmajorticklabels()
for label in xlabels:
    for i in range(cluster_num):
        if label.get_text() in color_label[i]:
            label.set_color(plt.cm.tab20(i))

plt.title(f'eps={eps} min_sample={min_sample}')
plt.savefig(f'__dendro_eps={eps}_min={min_sample}.png')
plt.clf()