"""
03_clustering.py — Phân cụm dữ liệu (1.3)
  - K-Means & GMM trên đặc trưng đầu vào
  - Đánh giá Silhouette & Davies–Bouldin
  - Trực quan hóa PCA(2)
  - Tạo nhãn phân cụm output (Baseline + Deviation)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

from config import RND, HORIZONS
from 01_preprocessing import run_preprocessing


# ==============================
# Cell A: Baseline + Deviation + Labeling (Semantic)
# ==============================

def build_output_labels(dfs):
    """Tạo nhãn baseline, deviation và semantic quantile labels."""
    H = 6
    D = dfs[H].copy().reset_index(drop=True)

    # 1) Baseline tuyến tính
    D["lat_baseline"] = D["lat_lag1"] + (D["lat_lag1"] - D["lat_lag2"])
    D["lon_baseline"] = D["lon_lag1"] + (D["lon_lag1"] - D["lon_lag2"])
    D = D.dropna(subset=["lat_baseline", "lon_baseline"]).reset_index(drop=True)

    # 2) Deviation
    D["lat_dev"] = D["lat_tgt"] - D["lat_baseline"]
    D["lon_dev"] = D["lon_tgt"] - D["lon_baseline"]

    # 3) Quantile labels có ngữ nghĩa
    def make_semantic_quantile_labels(series):
        q = pd.qcut(series, q=4)
        return pd.cut(
            series,
            bins=q.cat.categories,
            labels=["Neg_Large", "Neg_Small", "Pos_Small", "Pos_Large"],
            include_lowest=True
        )

    D["lat_label"] = make_semantic_quantile_labels(D["lat_dev"])
    D["lon_label"] = make_semantic_quantile_labels(D["lon_dev"])

    # 4) Output clustering labels
    out_features = D[["lat_dev", "lon_dev"]].to_numpy()

    km = KMeans(n_clusters=4, random_state=RND, n_init="auto")
    D["out_km_label"] = km.fit_predict(out_features).astype(str)

    gmm = GaussianMixture(n_components=4, random_state=RND)
    D["out_gmm_label"] = gmm.fit_predict(out_features).astype(str)

    # 5) Dev magnitude + cls_label
    D["dev_mag"] = np.sqrt(D["lat_dev"] ** 2 + D["lon_dev"] ** 2)

    class_names = ["Small_Error", "Medium_Small", "Medium_Large", "Large_Error"]
    D["cls_label"] = pd.qcut(D["dev_mag"], q=4, labels=class_names)

    print("Lat label distribution:")
    print(D["lat_label"].value_counts())
    print("\n=== Final classification label distribution ===")
    print(D["cls_label"].value_counts())

    return D


# ==============================
# Cell 1.3 — Clustering (K-Means & GMM) on INPUT features
# ==============================

def run_clustering(D_outlabels):
    drop_cols = [
        "lat_tgt", "lon_tgt",
        "lat_baseline", "lon_baseline",
        "lat_dev", "lon_dev",
        "lat_qlabel", "lon_qlabel",
        "out_km_label", "out_gmm_label"
    ]

    X_cluster = D_outlabels.select_dtypes(include=[np.number]) \
                           .drop(columns=[c for c in drop_cols if c in D_outlabels.columns],
                                 errors="ignore")

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imp.fit_transform(X_cluster)
    X_scaled = scaler.fit_transform(X_imp)
    print("Input shape for clustering:", X_scaled.shape)

    # PCA 6D
    pca = PCA(n_components=6, random_state=RND)
    X_pca = pca.fit_transform(X_scaled)

    print("Explained variance (6 PCs):",
          np.round(pca.explained_variance_ratio_ * 100, 2))
    print("Cumulative:",
          np.round(pca.explained_variance_ratio_.cumsum()[-1] * 100, 2), "%")

    # K-Means (k=4)
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=RND, n_init="auto")
    km_labels = kmeans.fit_predict(X_pca)

    # GMM (k=4)
    gmm = GaussianMixture(n_components=k, random_state=RND)
    gmm_labels = gmm.fit_predict(X_pca)

    # Đánh giá
    print("\n=== Clustering Quality Metrics ===")
    print("KMeans  - Silhouette:", silhouette_score(X_pca, km_labels))
    print("KMeans  - Davies–Bouldin:", davies_bouldin_score(X_pca, km_labels))
    print("GMM     - Silhouette:", silhouette_score(X_pca, gmm_labels))
    print("GMM     - Davies–Bouldin:", davies_bouldin_score(X_pca, gmm_labels))

    # Trực quan hóa PCA(2)
    pca2 = PCA(n_components=2, random_state=RND)
    X_pca2 = pca2.fit_transform(X_scaled)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_pca2[:, 0], y=X_pca2[:, 1],
                    hue=km_labels, palette="tab10", s=10, legend="full")
    plt.title("K-Means Clustering on PCA(2)")
    plt.xlabel("PC1"); plt.ylabel("PC2")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_pca2[:, 0], y=X_pca2[:, 1],
                    hue=gmm_labels, palette="tab10", s=10, legend="full")
    plt.title("GMM Clustering on PCA(2)")
    plt.xlabel("PC1"); plt.ylabel("PC2")

    plt.tight_layout()
    plt.show()

    # Lưu nhãn
    D_outlabels["cluster_kmeans"] = km_labels
    D_outlabels["cluster_gmm"] = gmm_labels

    print("\nCluster distribution (KMeans):")
    print(pd.Series(km_labels).value_counts().sort_index())
    print("\nCluster distribution (GMM):")
    print(pd.Series(gmm_labels).value_counts().sort_index())

    return D_outlabels


if __name__ == "__main__":
    _, dfs = run_preprocessing()
    D_outlabels = build_output_labels(dfs)
    D_outlabels = run_clustering(D_outlabels)
