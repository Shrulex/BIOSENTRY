# kmeans_clustering.py
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib

# Load preprocessed data
data = pd.read_csv('preprocessed_biomarker_data.csv')

# Initialize K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data)

# Save the K-Means model
joblib.dump(kmeans, 'models/kmeans_model.pkl')
print("K-Means model saved successfully as models/kmeans_model.pkl.")

# Visualize clusters
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=clusters, cmap='viridis')
plt.title("K-Means Clustering of Biomarker Data")
plt.savefig('results/cluster_visualization.png')
print("Cluster visualization saved as results/cluster_visualization.png.")
