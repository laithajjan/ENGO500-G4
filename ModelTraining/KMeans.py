import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Set the input directory and output directory
input_directory = "../Data/ProcessedData/"
output_directory = "../Models/"

# Read in the processed data
data = pd.read_csv(input_directory + "combined_data.csv", header=0)

# Assuming 'data' is your DataFrame
features = ['SOR', 'Ncg/steam']
target = 'CalDlyOil(m3/d)'

# Replace any zero values in 'scale' with a small positive value
data['scale'] = data['PrdHours(hr)']*data['NbrofWells']
data['scale'] = data['scale'].replace(0, 1e-8)

# Divide each feature by 'scale'
data[features] = data[features].div(data['scale'], axis=0)

data = data.dropna()

scaled_data = data[features]


# # Standardize the data (important for clustering and t-SNE)
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(data[features])
scaled_data = data[features]
scaled_data = scaled_data.dropna()

# Perform t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
tsne_results = tsne.fit_transform(scaled_data)

# Create a new column to indicate whether a data point has NCG or not (1: has NCG, 0: no NCG)
data['has_ncg'] = (data['CalInjGas(E3m3/d)'] > 0).astype(int)

# Visualize the data points with colors based on the presence of NCG
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=data['has_ncg'], cmap='viridis')

# Add a legend
unique_labels = np.unique(data['has_ncg'])
for label in unique_labels:
    plt.scatter([], [], c='k', alpha=0.3, label=f'{"Has NCG" if label == 1 else "No NCG"}')
plt.legend(title='NCG Status')

plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization with NCG Status')
plt.show()

# Perform K-means clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(scaled_data)
data['cluster'] = kmeans.labels_

# Set up a dictionary to map NCG status to marker shapes
ncg_marker_dict = {0: 'o', 1: 's'}

# Set up a dictionary to map cluster labels to colors
color_dict = {0: 'blue', 1: 'green', 2: 'red', 3: 'purple'}

# Create a scatter plot for each cluster and NCG status combination
for cluster in range(n_clusters):
    for ncg_status in [0, 1]:
        cluster_ncg_data = data[(data['cluster'] == cluster) & (data['has_ncg'] == ncg_status)]
        plt.scatter(tsne_results[cluster_ncg_data.index, 0], tsne_results[cluster_ncg_data.index, 1],
                    c=color_dict[cluster], marker=ncg_marker_dict[ncg_status], alpha=0.5, label=f'Cluster {cluster}, {"Has NCG" if ncg_status == 1 else "No NCG"}')

# Create a legend
plt.legend(title='Cluster and NCG Status')

# Set axis labels and title
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization with Cluster Labels and NCG Status')

# Show the plot
plt.show()
