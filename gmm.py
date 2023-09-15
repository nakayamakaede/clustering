from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
import pandas as pd
from sklearn.mixture import GaussianMixture
from matplotlib import cm

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
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
pos = mds.fit_transform(D)

res = pd.DataFrame(pos, columns=['x', 'y'])
"""

cluster_num=6
pred = GaussianMixture(n_components=cluster_num).fit_predict(D)

color_label=[]
for i in range(cluster_num):
    color_label.append([])

for i, j  in enumerate(pred):
    color_label[j].append(data_list[i])


Z = linkage(cc_dist, method='ward', metric='euclidean')
dendrogram(Z,labels=data_list)

ax = plt.gca()
xlabels = ax.get_xmajorticklabels()
for label in xlabels:
    for i in range(cluster_num):
        if label.get_text() in color_label[i]:
            label.set_color(plt.cm.tab20(i))

plt.title(f'GMM_cluster={cluster_num}')
plt.savefig(f'GMM_cluster={cluster_num}_3.png')
plt.clf()