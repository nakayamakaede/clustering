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

ori_label=[]
with open('data_list2.dat','r')as f:
    lines = f.readlines()
for line in lines:
    ori_label.append(str(line))

D = squareform(cc_dist)

distortions = []

for i  in range(1,11):                # 1~10クラスタまで一気に計算 
    km = KMeans(n_clusters=i,
                init='k-means++',     # k-means++法によりクラスタ中心を選択
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(D)                         # クラスタリングの計算を実行
    distortions.append(km.inertia_)   # km.fitするとkm.inertia_が得られる

plt.plot(range(1,11),distortions,marker='o')
plt.xlabel('Number of clusters',fontsize=18)
plt.ylabel('Distortion',fontsize=18)
plt.title('kmeans elbow', fontsize=20)
plt.savefig('kmeans_elbo.png')
plt.show()