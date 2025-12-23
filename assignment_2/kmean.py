import numpy as np
from collections import Counter
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


def calc_accuracy(cluster_labels, class_labels):
    cluster_stats = {}

    data = np.column_stack((cluster_labels, class_labels))

    for cluster_id, class_label in data:
        if cluster_id not in cluster_stats:
            # Initialize a Counter for each unique cluster ID
            cluster_stats[cluster_id] = Counter()
        cluster_stats[cluster_id][class_label] += 1

    # Process results
    final_results = []
    for cluster_id, counts in cluster_stats.items():
        # Most frequent class label and its count
        assigned_label, match_count = counts.most_common(1)[0]
        total_points = sum(counts.values())

        final_results.append(
            {
                "cluster_id": cluster_id,
                "assigned_label": assigned_label,
                "total_points": total_points,
                "match_count": match_count,
                "accuracy": match_count / total_points,
            }
        )

    total = sum(item["accuracy"] for item in final_results)
    mean = total / len(final_results) if final_results else 0

    return mean


def Cosine_IterativeKMean(
    features,
    y_true,
    k=10,
    max_iter=20,
    include_cost=True,
    include_accuracy=True,
    include_silhouette=True,
):
    results = {"cost": [], "accuracy": [], "silhouette": []}

    X = features
    X = normalize(X)
    n_samples, n_features = X.shape

    # initialize the centroids of the cluster, pick random points of the available samples
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]

    for _ in range(max_iter):
        # Calculate Cosine Distance (1 - dot product)
        similarities = np.dot(X, centroids.T)
        distances = 1 - similarities
        labels = np.argmin(distances, axis=1)

        # Calculate Cost (Sum of Cosine Distances)
        if include_cost:
            closest_distances = np.min(distances, axis=1)
            cost = np.sum(closest_distances)
            results["cost"].append(cost)

        # Calculate accuracy
        if include_accuracy:
            results["accuracy"].append(calc_accuracy(labels, y_true))

        # Calculate Silhouette Coefficient
        if include_silhouette:
            if len(np.unique(labels)) > 1:
                sil = silhouette_score(X, labels, sample_size=10000)
            else:
                sil = 0
            results["silhouette"].append(sil)

        # Update Centroids
        new_centroids = np.zeros((k, n_features))
        for c in range(k):
            points = X[labels == c]
            if len(points) > 0:
                mean_vec = points.mean(axis=0)
                # Re-normalize the new centroid
                norm_val = np.linalg.norm(mean_vec)
                if norm_val > 0:
                    new_centroids[c] = mean_vec / norm_val
                else:
                    new_centroids[c] = mean_vec
            else:
                new_centroids[c] = centroids[c]

        centroids = new_centroids

    return results


def Euclidean_IterativeKMean(
    features,
    y_true,
    k=10,
    max_iter=20,
    include_cost=True,
    include_accuracy=True,
    include_silhouette=True,
):
    results = {"cost": [], "accuracy": [], "silhouette": []}

    X = features
    n_samples, n_features = X.shape

    # initialize the centroids of the cluster, pick random points of the available samples
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]

    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Calculate Cost (SSE)
        if include_cost:
            closest_distances = np.min(distances, axis=1)
            cost = np.sum(closest_distances**2)
            results["cost"].append(cost)

        # Calculate accuracy
        if include_accuracy:
            results["accuracy"].append(calc_accuracy(labels, y_true))

        # Calculate Silhouette Coefficient
        if include_silhouette:
            if len(np.unique(labels)) > 1:
                sil = silhouette_score(X, labels, sample_size=10000)
            else:
                sil = 0
            results["silhouette"].append(sil)

        # Update Centroids
        new_centroids = np.zeros((k, n_features))
        for c in range(k):
            points = X[labels == c]
            if len(points) > 0:
                new_centroids[c] = points.mean(axis=0)
            else:
                new_centroids[c] = centroids[c]
        centroids = new_centroids
    return results
