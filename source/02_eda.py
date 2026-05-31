"""
02_eda.py — Phân tích và trực quan hóa dữ liệu
  - Tương quan đặc trưng với đầu ra
  - PCA visualization
  - Đánh giá PCA & LDA theo số chiều
  - Thống kê sau chuẩn hóa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances

from config import HORIZONS, RND
from 01_preprocessing import run_preprocessing, add_dynamic_features


# ==============================
# 8) Trực quan hóa mối quan hệ giữa đặc trưng chính và đầu ra
# ==============================

def plot_feature_correlation(dfs):
    H = 6
    D = dfs[H].sample(30000, random_state=42).copy()

    try:
        D = add_dynamic_features(D)
    except Exception:
        pass

    num_cols = [c for c in D.select_dtypes("number").columns if c not in ("lat_tgt", "lon_tgt")]

    corr_lat = D[num_cols + ["lat_tgt"]].corr()["lat_tgt"].drop("lat_tgt").sort_values(ascending=False)
    corr_lon = D[num_cols + ["lon_tgt"]].corr()["lon_tgt"].drop("lon_tgt").sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x=corr_lat.head(10), y=corr_lat.head(10).index, ax=axes[0], color="skyblue")
    axes[0].set_title("🔹 Tương quan với vĩ độ dự đoán (lat_tgt)")
    sns.barplot(x=corr_lon.head(10), y=corr_lon.head(10).index, ax=axes[1], color="lightgreen")
    axes[1].set_title("🔸 Tương quan với kinh độ dự đoán (lon_tgt)")
    plt.tight_layout()
    plt.show()

    feat_lat = corr_lat.index[0]
    feat_lon = corr_lon.index[0]

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=D[feat_lat], y=D["lat_tgt"], alpha=0.4, color="blue")
    plt.title(f"Quan hệ tuyến tính giữa {feat_lat} và lat_tgt")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=D[feat_lon], y=D["lon_tgt"], alpha=0.4, color="green")
    plt.title(f"Quan hệ tuyến tính giữa {feat_lon} và lon_tgt")
    plt.show()


# ==============================
# 9) PCA visualization
# ==============================

def plot_pca_visualization(dfs):
    H = 6
    D = dfs[H].sample(30000, random_state=42).copy()

    num_cols = [c for c in D.select_dtypes("number").columns if c not in ("lat_tgt", "lon_tgt")]
    X = D[num_cols].fillna(D[num_cols].median())
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=6, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(6)])
    pca_df["lat_tgt"] = D["lat_tgt"].values
    pca_df["lon_tgt"] = D["lon_tgt"].values

    explained = np.round(pca.explained_variance_ratio_ * 100, 2)
    print("📊 Phương sai giải thích của 6 thành phần đầu tiên:", explained)
    print("Tổng cộng:", explained.sum().round(2), "%")

    pairs = [(0, 1), (1, 2), (2, 3)]
    plt.figure(figsize=(15, 10))
    for i, (a, b) in enumerate(pairs, 1):
        plt.subplot(2, 2, i)
        sns.scatterplot(x=pca_df[f"PC{a+1}"], y=pca_df[f"PC{b+1}"],
                        hue=pca_df["lat_tgt"], palette="viridis", alpha=0.5, legend=False)
        plt.title(f"PC{a+1} vs PC{b+1}")
    plt.tight_layout()
    plt.show()


# ==============================
# Đánh giá PCA & LDA theo số chiều
# ==============================

def evaluate_pca_lda(dfs):
    D = dfs[HORIZONS[0]]
    X = D.select_dtypes("number").drop(columns=["lat_tgt", "lon_tgt"], errors="ignore")

    Xn = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]).fit_transform(X)

    labels = KMeans(n_clusters=4, random_state=42, n_init="auto").fit_predict(Xn)

    print("===== PCA: Explained variance =====")
    for k in [2, 4, 6, 8, 12]:
        pca = PCA(n_components=k, random_state=42)
        pca.fit(Xn)
        print(f"k={k:2d} → {pca.explained_variance_ratio_.sum():.2%}")

    print("\n===== LDA: Fisher score =====")
    for k in [1, 2, 3]:
        lda = LDA(n_components=k)
        Xl = lda.fit_transform(Xn, labels)
        centroids = np.vstack([Xl[labels == c].mean(0) for c in np.unique(labels)])
        fisher = pairwise_distances(centroids).mean() / np.mean(
            [np.var(Xl[labels == c], axis=0).sum() for c in np.unique(labels)]
        )
        print(f"k={k} → Fisher = {fisher:.4f}")

    return Xn, labels


def plot_lda_distribution(Xn, labels):
    """Phân bố dữ liệu theo LDA (1D)."""
    lda_1d = LDA(n_components=1)
    Xl_1d = lda_1d.fit_transform(Xn, labels)

    plt.figure(figsize=(8, 5))
    for c in np.unique(labels):
        sns.histplot(
            Xl_1d[labels == c, 0],
            bins=30,
            stat="count",
            alpha=0.6,
            label=f"Class {c}"
        )
    plt.xlabel("LDA Component 1")
    plt.ylabel("Tần suất")
    plt.title("Phân bố dữ liệu theo thành phần LDA")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==============================
# Mô tả dữ liệu sau khi chuẩn hóa
# ==============================

def describe_normalized(dfs):
    example_h = HORIZONS[0]
    D = dfs[example_h].copy()

    num_cols = [c for c in D.select_dtypes("number").columns if c not in ("lat_tgt", "lon_tgt")]

    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()

    X_imp = pd.DataFrame(imp.fit_transform(D[num_cols]), columns=num_cols)
    X_scaled = pd.DataFrame(sc.fit_transform(X_imp), columns=num_cols)

    desc = X_scaled.describe().T[['mean', 'std', 'min', 'max']].round(2)
    desc = desc.rename(columns={
        'mean': 'Giá trị trung bình',
        'std': 'Độ lệch chuẩn',
        'min': 'Nhỏ nhất',
        'max': 'Lớn nhất'
    })
    print("📊 Thống kê dữ liệu sau chuẩn hóa (StandardScaler):")
    print(desc)
    print(f"\nTổng số đặc trưng sau chuẩn hóa: {X_scaled.shape[1]}")
    print(f"Số bản ghi: {X_scaled.shape[0]}")

    print("\n🧩 Nhận xét:")
    print("- Tất cả các cột đều có mean ≈ 0 và std ≈ 1 → Chuẩn hóa thành công.")
    print("- Các giá trị min/max thể hiện mức độ phân tán sau chuẩn hóa.")
    print("- Bộ dữ liệu hiện sẵn sàng cho PCA hoặc huấn luyện mô hình hồi quy.")


if __name__ == "__main__":
    _, dfs = run_preprocessing()
    plot_feature_correlation(dfs)
    plot_pca_visualization(dfs)
    Xn, labels = evaluate_pca_lda(dfs)
    plot_lda_distribution(Xn, labels)
    describe_normalized(dfs)
