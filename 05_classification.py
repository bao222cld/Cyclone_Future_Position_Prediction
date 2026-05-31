"""
05_classification.py — Bài toán phân loại (1.3c)
  - Chuẩn bị X, y (quantile labels)
  - Naive Bayes & Random Forest với 3 tỷ lệ train/test
  - Đánh giá: F1-macro, confusion matrix
  - Random Forest & Naive Bayes với dữ liệu giảm chiều (PCA)
  - XGBoost (thử nghiệm)
  - Overfitting check, data leakage check
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)

from config import RND
from 01_preprocessing import run_preprocessing
from 03_clustering import build_output_labels


# ==============================
# Cell B: PCA + Clustering trên Input Features (cho phân loại)
# ==============================

def prepare_classification_data(D_outlabels):
    """Chuẩn bị X, y cho bài toán phân loại."""
    from sklearn.impute import SimpleImputer
    from sklearn.cluster import KMeans

    drop_cols = [
        "lat_tgt", "lon_tgt",
        "lat_baseline", "lon_baseline",
        "lat_dev", "lon_dev",
        "lat_label", "lon_label",
        "out_km_label", "out_gmm_label"
    ]

    label_choice = "quantile"  # "quantile" hoặc "out_cluster"
    D = D_outlabels.copy()

    if label_choice == "quantile":
        y = D["lat_label"]
    else:
        y = D["out_km_label"]

    X = D.select_dtypes(include=[np.number]) \
         .drop(columns=[c for c in drop_cols if c in D.columns], errors="ignore")

    mask = ~y.isna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    groups = D.loc[mask, "storm_id"].reset_index(drop=True)

    print("Label distribution:")
    print(y.value_counts())

    return X, y, groups


# ==============================
# Naive Bayes & Random Forest — 3 tỷ lệ train/test
# ==============================

def run_classification_experiments(X, y, groups):
    """Chạy NB và RF với 3 tỷ lệ chia dữ liệu."""
    splits = {
        "4:1": 0.20,
        "7:3": 0.30,
        "6:4": 0.40
    }

    last_nb, last_rf, last_Xtr, last_Xte, last_ytr, last_yte, last_le = (
        None, None, None, None, None, None, None
    )

    for split_name, test_size in splits.items():
        print("\n" + "=" * 50)
        print(f" Train : Test = {split_name}")
        print("=" * 50)

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RND)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        le = LabelEncoder()
        ytr = le.fit_transform(y_train.astype(str))
        yte = le.transform(y_test.astype(str))
        class_names = le.classes_.tolist()

        nb = GaussianNB()
        rf = RandomForestClassifier(n_estimators=150, random_state=RND, n_jobs=-1)
        nb.fit(Xtr, ytr)
        rf.fit(Xtr, ytr)

        for name, model in [("Naive Bayes", nb), ("Random Forest", rf)]:
            pred = model.predict(Xte)
            print(f"\n{name}")
            print("F1-macro:", f1_score(yte, pred, average="macro"))
            print(classification_report(yte, pred, target_names=class_names, digits=4))

        cm_nb = confusion_matrix(yte, nb.predict(Xte))
        cm_rf = confusion_matrix(yte, rf.predict(Xte))

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Oranges",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Naive Bayes ({split_name})")
        plt.xlabel("Predicted label"); plt.ylabel("True label")

        plt.subplot(1, 2, 2)
        sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Random Forest ({split_name})")
        plt.xlabel("Predicted label"); plt.ylabel("True label")

        plt.tight_layout()
        plt.show()

        last_nb, last_rf = nb, rf
        last_Xtr, last_Xte = Xtr, Xte
        last_ytr, last_yte = ytr, yte
        last_le = le

    return last_nb, last_rf, last_Xtr, last_Xte, last_ytr, last_yte, last_le


# ==============================
# Summary
# ==============================

def print_summary():
    results = {
        "4:1": {"NB": 0.5741, "RF": 0.7109},
        "7:3": {"NB": 0.5738, "RF": 0.7086},
        "6:4": {"NB": 0.5745, "RF": 0.7041}
    }

    print("Quick Observations:")
    print("-" * 50)
    for split, metrics in results.items():
        print(f"Train:Test = {split}")
        print(f"  - Naive Bayes F1-macro: {metrics['NB']:.4f}")
        print(f"  - Random Forest F1-macro: {metrics['RF']:.4f}")
        if metrics['RF'] > metrics['NB']:
            print("    => RF consistently outperforms NB across splits.")
        print()

    print("Overall:")
    print(" - F1-macro của NB ổn định ~0.574, không thay đổi nhiều theo split.")
    print(" - RF cho F1-macro cao hơn NB (~0.705–0.711), hơi giảm khi test size tăng.")
    print(" - Split nhỏ hơn (4:1) hơi tốt hơn cho RF, nhưng sự khác biệt không lớn.")


# ==============================
# PCA + Classification
# ==============================

def run_pca_classification(Xtr, Xte, ytr, yte):
    """RF và NB trên dữ liệu giảm chiều bằng PCA."""
    k = max(1, Xtr.shape[1] // 3)

    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr)
    Xte_scaled = scaler.transform(Xte)

    pca = PCA(n_components=k, random_state=RND)
    Xtr_p = pca.fit_transform(Xtr_scaled)
    Xte_p = pca.transform(Xte_scaled)

    print("PCA cumulative variance:",
          np.round(pca.explained_variance_ratio_.cumsum()[-1] * 100, 2), "%")

    rf_p = RandomForestClassifier(n_estimators=150, random_state=RND, n_jobs=-1)
    rf_p.fit(Xtr_p, ytr)
    pred_p = rf_p.predict(Xte_p)
    print("RF PCA F1-macro:", f1_score(yte, pred_p, average="macro"))

    nb_p = GaussianNB()
    nb_p.fit(Xtr_p, ytr)
    pred_nb_p = nb_p.predict(Xte_p)
    print("NB PCA F1-macro:", f1_score(yte, pred_nb_p, average="macro"))

    return rf_p, nb_p, Xtr_p, Xte_p


def plot_pca_comparison():
    """Biểu đồ so sánh F1 With vs Without PCA."""
    splits_labels = ["4:1", "7:3", "6:4"]
    nb_no_pca = [0.5624, 0.5700, 0.5692]
    rf_no_pca = [0.6870, 0.6903, 0.6900]
    nb_pca = [0.5292] * 3
    rf_pca = [0.6131] * 3

    x = np.arange(len(splits_labels))
    width = 0.35

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.bar(x - width / 2, nb_no_pca, width, label="Naive Bayes")
    plt.bar(x + width / 2, rf_no_pca, width, label="Random Forest")
    plt.xticks(x, splits_labels)
    plt.ylim(0.5, 0.75)
    plt.title("Without PCA (Group Split)")
    plt.ylabel("F1-macro")
    plt.xlabel("Train : Test split")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.bar(x - width / 2, nb_pca, width, label="Naive Bayes")
    plt.bar(x + width / 2, rf_pca, width, label="Random Forest")
    plt.xticks(x, splits_labels)
    plt.ylim(0.5, 0.75)
    plt.title("With PCA (~92.8% variance retained)")
    plt.xlabel("Train : Test split")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.suptitle("F1-macro Comparison of Models (Group-aware Split)", fontsize=13)
    plt.tight_layout()
    plt.show()


# ==============================
# XGBoost (Thử nghiệm)
# ==============================

def run_xgboost(Xtr, Xte, ytr, yte, le):
    """Thử nghiệm XGBoost Classifier."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("XGBoost chưa được cài đặt. Chạy: pip install xgboost")
        return None

    class_names = [str(c) for c in le.classes_]

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=len(class_names),
        eval_metric="mlogloss",
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=RND,
        n_jobs=-1
    )
    xgb.fit(Xtr, ytr)
    pred = xgb.predict(Xte)

    print("\nXGBoost")
    print("F1-macro:", f1_score(yte, pred, average="macro"))
    print(classification_report(yte, pred, target_names=class_names))

    cm = confusion_matrix(yte, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix — XGBoost")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.show()

    return xgb


def plot_model_comparison(nb, rf, xgb, Xtr, Xte, ytr, yte):
    """Train vs Test F1, Accuracy vs F1-macro, confidence distribution."""
    models = {"Naive Bayes": nb, "Random Forest": rf}
    if xgb is not None:
        models["XGBoost"] = xgb

    train_scores = []
    test_scores = []
    for name, model in models.items():
        train_scores.append(f1_score(ytr, model.predict(Xtr), average="macro"))
        test_scores.append(f1_score(yte, model.predict(Xte), average="macro"))

    # Train vs Test
    plt.figure(figsize=(8, 5))
    plt.bar(models.keys(), train_scores, alpha=0.7, label="Train")
    plt.bar(models.keys(), test_scores, alpha=0.7, label="Test")
    plt.ylabel("F1-macro")
    plt.title("Train vs Test Performance (Overfitting Analysis)")
    plt.legend()
    plt.ylim(0.5, 0.8)
    plt.show()

    # Accuracy vs F1
    summary = pd.DataFrame({
        "Model": list(models.keys()),
        "Accuracy": [accuracy_score(yte, m.predict(Xte)) for m in models.values()],
        "F1_macro": test_scores
    })
    summary.set_index("Model").plot(kind="bar", figsize=(8, 5), rot=0)
    plt.title("Model Effectiveness Comparison")
    plt.ylabel("Score")
    plt.ylim(0.5, 0.8)
    plt.grid(axis="y", alpha=0.3)
    plt.show()

    # Confidence distribution (XGBoost)
    if xgb is not None:
        proba = xgb.predict_proba(Xte)
        max_proba = proba.max(axis=1)
        plt.figure(figsize=(7, 4))
        plt.hist(max_proba, bins=30, edgecolor="black")
        plt.xlabel("Max predicted probability")
        plt.ylabel("Count")
        plt.title("Prediction Confidence Distribution (XGBoost)")
        plt.show()


# ==============================
# Data Leakage Check
# ==============================

def data_leakage_check(X, y):
    forbidden_cols = [
        "lat_tgt", "lon_tgt",
        "lat_dev", "lon_dev",
        "dev_mag",
        "lat_label", "lon_label", "cls_label",
        "out_km_label", "out_gmm_label"
    ]

    print("=== DATA LEAKAGE SANITY CHECK ===\n")
    print("Forbidden columns present in X:")
    leak_found = False
    for col in forbidden_cols:
        present = col in X.columns
        print(f"{col:15s}: {present}")
        if present:
            leak_found = True

    print("\n=== RESULT ===")
    if leak_found:
        print("Potential DATA LEAKAGE detected! Check feature construction.")
    else:
        print("No data leakage detected. Feature set is clean.")

    print("\nFeature count:", X.shape[1])
    print("Label used:", y.name if hasattr(y, "name") else "encoded label")


if __name__ == "__main__":
    _, dfs = run_preprocessing()
    D_outlabels = build_output_labels(dfs)

    X, y, groups = prepare_classification_data(D_outlabels)
    data_leakage_check(X, y)

    nb, rf, Xtr, Xte, ytr, yte, le = run_classification_experiments(X, y, groups)
    print_summary()
    run_pca_classification(Xtr, Xte, ytr, yte)
    plot_pca_comparison()

    xgb = run_xgboost(Xtr, Xte, ytr, yte, le)
    plot_model_comparison(nb, rf, xgb, Xtr, Xte, ytr, yte)
