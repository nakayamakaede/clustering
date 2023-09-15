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
eps = [x/100 for x in range(1,101)]
min_sample=7
y = []
for i in eps:
    db = DBSCAN(eps=i, min_samples=min_sample)

    dbscan = db.fit_predict(D)
    labels = db.labels_
    
    y.append(len(set(labels)))


plt.plot(eps, y)
plt.xlabel('eps',fontsize=20)
plt.ylabel('number of cluster',fontsize=20)
plt.title(f'min_samples={min_sample}',fontsize=20)
plt.savefig(f'__dendro_min={min_sample}.png')
plt.show()

for i, j in enumerate(y):
    if j == max(y):
        print(eps[i])
