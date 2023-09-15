from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
import pandas as pd
from sklearn.cluster import KMeans

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

km = KMeans(n_clusters=5,            # クラスターの個数
            init='k-means++',        # セントロイドの初期値をランダムに設定
            n_init=10,               # 異なるセントロイドの初期値を用いたk-meansあるゴリmズムの実行回数
            max_iter=300,            # k-meansアルゴリズムの内部の最大イテレーション回数
            tol=1e-04,               # 収束と判定するための相対的な許容誤差
            random_state=0)          # セントロイドの初期化に用いる乱数発生器の状態
y_km = km.fit_predict(D)

#kmeans_color
cluster_num = len(set(y_km))

color_label=[]
for i in range(cluster_num):
    color_label.append([])
labels_plus= y_km - 1

for i, j  in enumerate(labels_plus):
    color_label[j].append(data_list[i])

#dendrogram2
Z = linkage(cc_dist, method='ward', metric='euclidean')
dendrogram(Z,labels=data_list)

ax = plt.gca()
xlabels = ax.get_xmajorticklabels()
for label in xlabels:
    for i in range(cluster_num):
        if label.get_text() in color_label[i]:
            label.set_color(plt.cm.tab20(i))

plt.title(f'kmeans  number of cluster={cluster_num}')
plt.savefig(f'kmeans_number_of_cluster={cluster_num}.png')
plt.clf()
