import numpy as np
import random
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

def pam(X, k, max_iter=100):
    n = X.shape[0]
    distance_matrix = pairwise_distances(X, metric='euclidean')
    current_medoids = random.sample(range(n), k)

    for _ in range(max_iter):
        labels = np.argmin(distance_matrix[:, current_medoids], axis=1)
        best_medoids = current_medoids[:]
        best_cost = compute_total_cost(distance_matrix, current_medoids, labels)

        for i in range(n):
            if i in current_medoids:
                continue
            for m in range(k):
                temp_medoids = current_medoids[:]
                temp_medoids[m] = i
                temp_labels = np.argmin(distance_matrix[:, temp_medoids], axis=1)
                temp_cost = compute_total_cost(distance_matrix, temp_medoids, temp_labels)

                if temp_cost < best_cost:
                    best_cost = temp_cost
                    best_medoids = temp_medoids[:]

        if set(best_medoids) == set(current_medoids):
            break
        current_medoids = best_medoids

    final_labels = np.argmin(distance_matrix[:, current_medoids], axis=1)
    return current_medoids, final_labels

def compute_total_cost(distance_matrix, medoids, labels):
    return sum(distance_matrix[i, medoids[labels[i]]] for i in range(len(labels)))

if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    k = 3
    medoids, labels = pam(X, k)

    plt.figure(figsize=(8, 6))
    for i in range(k):
        cluster_points = X[np.array(labels) == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i+1}")
        plt.scatter(X[medoids[i], 0], X[medoids[i], 1], s=200, c='black', marker='X', edgecolor='white', linewidth=2)

    plt.title("PAM Clustering Result")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()