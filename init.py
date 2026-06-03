#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pandas import DataFrame
from sklearn.cluster import kmeans_plusplus as sklearn_kmeans_plusplus


def kmeans_plusplus(X: DataFrame, n_clusters: int, random_state: int) -> DataFrame:
    assert len(X.drop_duplicates()) >= n_clusters > 0

    print("Initialising centroids using k-means++")

    centroids, indices = sklearn_kmeans_plusplus(
        X=X.to_numpy(),
        n_clusters=n_clusters,
        random_state=random_state
    )

    centroids = DataFrame(centroids, columns=X.columns)
    centroids.index.name = 'cluster'

    print(f"Centroids shape: {centroids.shape}")

    return centroids