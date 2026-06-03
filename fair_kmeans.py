import numpy as np
import pandas as pd


# ---------------------------
# KL DIVERGENCE
# ---------------------------
def compute_kl(Px, Pc):
    Pc = Pc.replace(0, 1e-10)
    return np.sum(Px * np.log(Px / Pc))


# ---------------------------
# DISTRIBUTION
# ---------------------------
def get_distribution(series):
    return series.value_counts(normalize=True).sort_index()


# ---------------------------
# FAIR ASSIGNMENT
# ---------------------------
def fair_assignment(X, s, centroids, lambda_):

    X_np = X.to_numpy()
    centroids_np = centroids.to_numpy()

    n = len(X)
    k = len(centroids)

    # overall distribution
    Px = get_distribution(s)

    # temporary random assignment
    c = pd.Series(np.random.randint(0, k, size=n), index=X.index)

    # compute cluster distributions
    cluster_dist = {}
    for j in range(k):
        cluster_s = s[c == j]
        if len(cluster_s) == 0:
            cluster_dist[j] = Px
        else:
            cluster_dist[j] = get_distribution(cluster_s).reindex(Px.index, fill_value=1e-10)

    new_c = []

    # assign each point
    for i, x in enumerate(X_np):

        costs = []

        for j in range(k):

            # distance
            dist = np.sum((x - centroids_np[j]) ** 2)

            # KL fairness
            Pc = cluster_dist[j]
            kl = compute_kl(Px, Pc)

            cost = dist + lambda_ * kl
            costs.append(cost)

        new_c.append(np.argmin(costs))

    return pd.Series(new_c, index=X.index)

# ---------------------------
# UPDATE CENTROIDS
# ---------------------------
def update_centroids(X, c, k):

    centroids = []

    for j in range(k):
        cluster_points = X[c == j]

        if len(cluster_points) == 0:
            # if empty cluster, keep random point
            centroids.append(X.sample(1).iloc[0].values)
        else:
            centroids.append(cluster_points.mean().values)

    return pd.DataFrame(centroids, columns=X.columns)


# ---------------------------
# FULL FAIR KMEANS
# ---------------------------
def fair_kmeans(X, s, centroids, k, lambda_, max_iter=10, tol=1e-4):

    prev_c = None

    for it in range(max_iter):

        print(f"\nIteration {it+1}")

        # assignment
        c = fair_assignment(X, s, centroids, lambda_)

        # stopping condition (if assignments don't change)
        if prev_c is not None and c.equals(prev_c):
            print("Converged early!")
            break

        prev_c = c.copy()

        # update
        centroids = update_centroids(X, c, k)

    return c, centroids

# ---------------------------
# UTILITY (SSE)
# ---------------------------
def compute_utility(X, c, centroids):

    X_np = X.to_numpy()
    centroids_np = centroids.to_numpy()

    total = 0

    for i, cluster in enumerate(c):
        total += np.sum((X_np[i] - centroids_np[cluster]) ** 2)

    return total


# ---------------------------
# FAIRNESS (KL)
# ---------------------------
def compute_fairness(s, c):

    Px = s.value_counts(normalize=True).sort_index()

    k = c.nunique()
    total_kl = 0

    for j in range(k):
        cluster_s = s[c == j]

        if len(cluster_s) == 0:
            continue

        Pc = cluster_s.value_counts(normalize=True).reindex(Px.index, fill_value=1e-10)

        Pc = Pc.replace(0, 1e-10)
        kl = np.sum(Px * np.log(Px / Pc))

        total_kl += kl

    return total_kl / k