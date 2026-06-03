# ============================
# FINAL FAIR KMEANS (FAST + PROGRESS + CLEAN)
# ============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from src.init import kmeans_plusplus
from src.fair_kmeans import compute_utility

# ---------------------------
# CREATE FOLDERS
# ---------------------------
os.makedirs("outputs/csvs", exist_ok=True)
os.makedirs("outputs/pca_plots", exist_ok=True)
os.makedirs("outputs/graphs", exist_ok=True)

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv("data/adult.csv", sep="\t")

s = df["age"].values
X = df.drop(columns=["age", "object"]).values

# ---------------------------
# FIND BEST k
# ---------------------------
k_values = list(range(2, 10))
sil_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)
    sil_scores.append(silhouette_score(X, labels))

k_final = k_values[np.argmax(sil_scores)]
print(f"\n✅ Best k selected: {k_final}")

# ---------------------------
# PRECOMPUTE
# ---------------------------
bins = np.linspace(s.min(), s.max(), 20)

def kl_divergence(p, q):
    eps = 1e-10
    return np.sum((p + eps) * np.log((p + eps) / (q + eps)))

def get_distribution(values):
    hist, _ = np.histogram(values, bins=bins, density=True)
    return hist / (hist.sum() + 1e-10)

global_dist = get_distribution(s)

# ---------------------------
# FAIR KMEANS
# ---------------------------
def fair_kmeans_full(X, s, centroids, k, lambda_, max_iter=20):

    distances = np.linalg.norm(X[:, None] - centroids, axis=2)
    c = np.argmin(distances, axis=1)

    for it in range(max_iter):

        # update centroids
        for i in range(k):
            pts = X[c == i]
            if len(pts) > 0:
                centroids[i] = pts.mean(axis=0)

        new_c = c.copy()

        for i in range(len(X)):

            best_cluster = c[i]
            best_score = float('inf')

            for cluster_id in range(k):

                temp_c = new_c.copy()
                temp_c[i] = cluster_id

                dist = np.linalg.norm(X[i] - centroids[cluster_id])

                fairness = 0
                for j in range(k):
                    vals = s[temp_c == j]
                    if len(vals) == 0:
                        continue
                    fairness += kl_divergence(get_distribution(vals), global_dist)

                score = dist + lambda_ * 50 * fairness

                if score < best_score:
                    best_score = score
                    best_cluster = cluster_id

            new_c[i] = best_cluster

        utility = compute_utility(pd.DataFrame(X), pd.Series(new_c), pd.DataFrame(centroids))

        fairness_total = 0
        for j in range(k):
            vals = s[new_c == j]
            if len(vals) == 0:
                continue
            fairness_total += kl_divergence(get_distribution(vals), global_dist)

        objective = utility + lambda_ * fairness_total

        print(f"   Iter {it+1:02d} → U: {utility:.1f} | F: {fairness_total:.4f} | Obj: {objective:.1f}")

        if np.array_equal(new_c, c):
            print("   ✅ Converged")
            break

        c = new_c

    return c, centroids, utility, fairness_total


# ---------------------------
# RUN FOR LAMBDAS
# ---------------------------
lambda_values = np.arange(0, 6.1, 0.25)
runs_per_lambda = 3

results = []

print("\n🚀 Starting Fair K-Means...\n")

for lambda_ in tqdm(lambda_values, desc="Lambda Progress"):

    utilities_list = []
    fairness_list = []

    print(f"\n========== Lambda = {lambda_:.2f} ==========")

    for run in tqdm(range(runs_per_lambda), desc="Runs", leave=False):

        print(f"\n--- Run {run+1}/{runs_per_lambda} ---")

        centroids = kmeans_plusplus(
            pd.DataFrame(X), k_final, random_state=run
        ).values.copy()

        c, centroids, utility, fairness = fair_kmeans_full(
            X, s, centroids, k_final, lambda_
        )

        utilities_list.append(utility)
        fairness_list.append(fairness)

    avg_utility = np.mean(utilities_list)
    avg_fairness = np.mean(fairness_list)

    print(f"\n➡️ Avg U: {avg_utility:.2f} | Avg F: {avg_fairness:.4f}")

    results.append({
        "lambda": lambda_,
        "utility": avg_utility,
        "fairness": avg_fairness
    })

# ---------------------------
# RESULTS + BEST LAMBDA
# ---------------------------
results_df = pd.DataFrame(results)

# normalize
results_df["utility_norm"] = (
    (results_df["utility"] - results_df["utility"].min()) /
    (results_df["utility"].max() - results_df["utility"].min() + 1e-10)
)

results_df["fairness_norm"] = (
    (results_df["fairness"] - results_df["fairness"].min()) /
    (results_df["fairness"].max() - results_df["fairness"].min() + 1e-10)
)

results_df["score"] = results_df["utility_norm"] + results_df["fairness_norm"]

best_row = results_df.loc[results_df["score"].idxmin()]
best_lambda = best_row["lambda"]

print("\n🎯 BEST LAMBDA FOUND:")
print(f"👉 Lambda = {best_lambda:.2f}")
print(f"👉 Utility = {best_row['utility']:.2f}")
print(f"👉 Fairness = {best_row['fairness']:.4f}")

print("\n🎉 DONE SUCCESSFULLY 🚀")

# ============================
# NORMAL KMEANS (NO FAIRNESS)
# ============================

print("\n🚀 Running Normal K-Means (No Fairness)...")

kmeans_normal = KMeans(n_clusters=k_final, random_state=0, n_init=10)
labels_normal = kmeans_normal.fit_predict(X)

# ---------------------------
# PLOT: AGE DISTRIBUTION COMPARISON
# ---------------------------
plt.figure(figsize=(8,6))

# dataset
plt.hist(s, bins=bins, density=True, histtype='step', linestyle='--', label='dataset')

# fair clustering (best lambda)
best_lambda = best_row["lambda"]

centroids = kmeans_plusplus(pd.DataFrame(X), k_final, random_state=0).values.copy()
labels_fair, _, _, _ = fair_kmeans_full(X, s, centroids, k_final, best_lambda)

for i in range(k_final):
    plt.hist(s[labels_fair == i], bins=bins, density=True, histtype='step', label=f'fair_cluster_{i}')

for i in range(k_final):
    plt.hist(s[labels_normal == i], bins=bins, density=True, histtype='step', linestyle=':', label=f'normal_cluster_{i}')

plt.xlabel("Age")
plt.ylabel("Proportion")
plt.title("Age Distribution: Fair vs Normal Clustering")
plt.legend()
plt.savefig("outputs/graphs/fair_vs_normal_age.png")
plt.show()


# ---------------------------
# PCA (k = 4, lambda = 2.5)
# ---------------------------
print("\n📊 Running PCA Visualization (k=4, lambda=2.5)...")

k_pca = 4
lambda_pca = 2.5

centroids = kmeans_plusplus(pd.DataFrame(X), k_pca, random_state=0).values.copy()
labels_pca, centroids, _, _ = fair_kmeans_full(X, s, centroids, k_pca, lambda_pca)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(7,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_pca, cmap='viridis', s=10)
plt.title("PCA Visualization (Fair K-Means, k=4, λ=2.5)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("outputs/pca_plots/pca_k4_lambda2_5.png")
plt.show()


# ---------------------------
# EXTRA ANALYSIS
# ---------------------------

print("\n📈 Extra Analysis...")

# Silhouette score comparison
sil_normal = silhouette_score(X, labels_normal)
sil_fair = silhouette_score(X, labels_fair)

print(f"👉 Silhouette (Normal): {sil_normal:.4f}")
print(f"👉 Silhouette (Fair):   {sil_fair:.4f}")

# Fairness vs Lambda plot
plt.figure(figsize=(7,5))
plt.plot(results_df["lambda"], results_df["fairness"], marker='o')
plt.xlabel("Lambda")
plt.ylabel("Fairness (KL Divergence)")
plt.title("Fairness vs Lambda")
plt.grid()
plt.savefig("outputs/graphs/fairness_vs_lambda.png")
plt.show()

# Utility vs Lambda plot
plt.figure(figsize=(7,5))
plt.plot(results_df["lambda"], results_df["utility"], marker='o')
plt.xlabel("Lambda")
plt.ylabel("Utility")
plt.title("Utility vs Lambda")
plt.grid()
plt.savefig("outputs/graphs/utility_vs_lambda.png")
plt.show()