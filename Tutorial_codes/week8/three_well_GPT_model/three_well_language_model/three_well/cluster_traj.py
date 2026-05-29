import numpy as np
import torch 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


data = np.loadtxt("../all_time_position.txt")

# Extract x and y
x = data[:, 0]
y = data[:, 1]






 



plt.figure(figsize=(6,5))
# 2D histogram
plt.hist2d(x, y, bins=100, cmap="inferno")


# Other Cmap options to see cool plot

#plt.hist2d(x, y, bins=100, cmap="rainbow")
#cmap="plasma"    # purple to yellow
#cmap="inferno"   # dark to fiery
#cmap="magma"     # dark purple to orange
#cmap="jet"       # rainbow


cbar = plt.colorbar()
cbar.set_label("Density", fontsize=12, fontweight='bold')

plt.xlabel("x", fontsize=12, fontweight='bold')
plt.ylabel("y", fontsize=12, fontweight='bold')
plt.title("Toy Threewell Potential", fontsize=14, fontweight='bold')

# make ticks bold
plt.xticks(fontsize=10, fontweight='bold')
plt.yticks(fontsize=10, fontweight='bold')

plt.savefig("three_well_potential.png", dpi=600, bbox_inches='tight')
plt.close()



from sklearn.cluster import KMeans


X = np.column_stack((x, y)) 
print(X.shape)

kmeans = KMeans(n_clusters=3,random_state=42)  
kmeans.fit(X)


# numeric labels
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# sort clusters by x-coordinate , this is to have a continuity

order = np.argsort(centers[:, 0])

# map ordered labels --> a, b, c
label_map = {order[0]: 'a', order[1]: 'b', order[2]: 'c'}
ordered_labels = np.vectorize(label_map.get)(labels)


print(ordered_labels)





plt.figure(figsize=(8, 6))

for name, color in zip(['a', 'b', 'c'], ['green', 'red', 'blue']):
    plt.scatter(
        x[ordered_labels == name],
        y[ordered_labels == name],
        color=color,
        label=name,
        s=5, alpha=0.6
    )

# centroids
plt.scatter(
    centers[:, 0], centers[:, 1],
    color='black', marker='x', s=120, linewidths=2,
    label='Centroids'
)

plt.xlabel("x(t)", fontsize=12, fontweight='bold')
plt.ylabel("y(t)", fontsize=12, fontweight='bold')
plt.title("K-Means Clustering of MD Trajectory", fontsize=14, fontweight='bold')

plt.xticks(fontsize=10, fontweight='bold')
plt.yticks(fontsize=10, fontweight='bold')

legend = plt.legend()
for text in legend.get_texts():
    text.set_fontweight('bold')

plt.savefig("clustered_traj.png", dpi=600, bbox_inches='tight')
plt.close()


np.savetxt("cluster_traj.txt", ordered_labels, fmt='%s')

print("Saved cluster labels to cluster_traj.txt")















