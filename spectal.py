from sklearn import cluster
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

cc_dist=[]
with open('cc2.dat','r')as f:
    lines = f.readlines()
for line in lines:
    cc_dist.append(float(line))

data_list=[]
with open('data_list2.dat','r')as f:
    lines = f.readlines()
for line in lines:
    data_list.append(str(line))

D = squareform(cc_dist)

spkm = cluster.SpectralClustering(n_clusters=4,affinity="nearest_neighbors")
res_spkm = spkm.fit(D)
#res_spkm.affinity_matrix_ # kernel変換後の行列
labels=res_spkm.labels_ #分類後のラベル

cluster_num = len(set(labels))
color_label=[]
for i in range(cluster_num):
    color_label.append([])

for i, j  in enumerate(labels):
    color_label[j].append(data_list[i])


Z = linkage(cc_dist, method='ward', metric='euclidean')
dendrogram(Z,labels=data_list)

ax = plt.gca()
xlabels = ax.get_xmajorticklabels()
for label in xlabels:
    for i in range(cluster_num):
        if label.get_text() in color_label[i]:
            label.set_color(plt.cm.tab20(i))
            
plt.title(f'nearest_neighbors')
plt.savefig(f'nearest_neighbors_{cluster_num}.png')
