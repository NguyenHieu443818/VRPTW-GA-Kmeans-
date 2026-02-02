import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans

# Generating synthetic time series data
np.random.seed(0)
# 10 time series, each of length 100
time_series_data = np.random.randn(10, 100)

# Extracting subsequences
window_size = 2
subsequences = [time_series_data[i, j:j+window_size]
                for i in range(time_series_data.shape[0])
                for j in range(time_series_data.shape[1] - window_size + 1)]
subsequences = np.array(subsequences)
print('subsequences', subsequences)

# Standardizing the subsequences
scaler = StandardScaler()
subsequences_scaled = scaler.fit_transform(subsequences)
print('subsequences_scaled', subsequences_scaled)
# Clustering using k-Means
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(subsequences_scaled)

# Display cluster labels for the first time series
print(labels[:time_series_data.shape[1] - window_size + 1])
