import random
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=2, tolerance=0.00001, max_iterations=300):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def fit(self, features):
        number_of_features = np.ma.size(features, axis=0)
        random_feature_indices = np.random.choice(range(number_of_features), size=self.k, replace=False)
        centroids = dict(enumerate(np.take(features, indices=random_feature_indices, axis=0).tolist()))
        for iteration in range(self.max_iterations):
            clusters = self.__cluster_features(features, centroids)
            previous_centroids = dict(centroids)
            for index in centroids:
                centroids[index] = np.mean(clusters[index], axis=0)
            if iteration > 0 and self.__is_optimized_fit(previous_centroids, centroids):
                break
        print("iterations:", iteration) 
        return (centroids, clusters)

    def __cluster_features(self, features, centroids):
        clusters = { group: [] for group in range(self.k) }
        for feature in features:
            centroid_distances = [self.__euclidean_distance(feature, centroid) for centroid in centroids.values()]
            closest_centroid_index = centroid_distances.index(min(centroid_distances))
            clusters[closest_centroid_index].append(feature)
        return clusters

    def __euclidean_distance(self, plot1, plot2):
        return np.linalg.norm(np.array(plot1) - np.array(plot2))

    def __is_optimized_fit(self, previous_centroids, current_centroids):
        optimized = True
        for index in current_centroids:
            relative_decrease = np.sum(np.absolute((current_centroids[index]-previous_centroids[index])/previous_centroids[index]*100.0))
            if relative_decrease > self.tolerance:
                optimized = False
        return optimized

def calculate_cluster_mean(cluster):
    dimensions = len(cluster[0])
    cluster_mean = []
    for d in range(dimensions):
        cluster_mean.append(np.mean(c[d] for c in cluster))
    return cluster_mean

# features = np.random.rand(250,2)
features = np.array([[5, 3], [10, 15], [15, 12], [24, 10], [30, 45], [85, 70], [71, 80], [60, 78], [55, 52],[80, 91]])

k_means = KMeans(k=2)
(centroids, clusters)  = k_means.fit(features)

for key in clusters:
    plt.scatter([f[0] for f in clusters[key]], [f[1] for f in clusters[key]])
plt.scatter([centroids[c][0] for c in centroids], [centroids[c][1] for c in centroids], s=2)
plt.show()