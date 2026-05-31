"""
04_regression.py — Phân tích hồi quy (1.4)
  - RidgeCV (sparse OHE, no PCA) + MLP (SVD-128, early stopping)
  - So sánh tỷ lệ train/test: 4:1, 7:3, 6:4
  - So sánh Original vs PCA
  - Kiểm tra Overfitting
  - Hiệu chỉnh Regularization MLP
  - Residual Analysis
"""

import numpy as np
import pandas as pd
import gc
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import RND, HORIZONS, SAMPLE_SIZE
from 01_preprocessing import run_preprocessing, add_dynamic_features

warnings.filterwarnings("ignore")


# ==============================
# Helpers
# ==============================

def haversine_vec(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


def geodesic_km_avg(y_true, y_pred):
    return float(np.nanmean(haversine_vec(
        y_true[:, 0], y_true[:, 1], y_pred[:, 0], y_pred[:, 1]
    )))


def _sanitize_categoricals(df_in, cat_cols):
    df = df_in.copy()
    for c in cat_cols:
        df[c] = df[c].astype(str)
        df[c] = df[c].replace("nan", "__MISSING__")
    return df


def make_ohe_sparse():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                             min_frequency=0.005, max_categories=64)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


# ==============================
# Regression Optimized
# RidgeCV (sparse, no PCA) + MLP (SVD-128)
# ==============================

def regression_optimized(D, test_size=0.3, svd_components=128):
    """Chạy RidgeCV và MLP trên một horizon dataset."""
    if SAMPLE_SIZE is not None and len(D) > SAMPLE_SIZE:
        D = D.sample(SAMPLE_SIZE, random_state=RND).reset_index(drop=True)

    D = add_dynamic_features(D)

    cat_cols = D.select_dtypes("object").columns.tolist()
    num_cols = [c for c in D.select_dtypes("number").columns if c not in ("lat_tgt", "lon_tgt")]

    D_cat = _sanitize_categoricals(D[cat_cols], cat_cols) if cat_cols else pd.DataFrame(index=D.index)
    X_all = pd.concat([D_cat, D[num_cols]], axis=1) if cat_cols else D[num_cols]
    y_all = D[["lat_tgt", "lon_tgt"]].values

    if "storm_id" in D.columns:
        groups = D["storm_id"].astype(str).values
    else:
        groups = np.arange(len(D))

    gss = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=RND)
    tr_idx, te_idx = next(gss.split(X_all, y_all, groups))
    Xtr, Xte = X_all.iloc[tr_idx], X_all.iloc[te_idx]
    ytr, yte = y_all[tr_idx], y_all[te_idx]

    # --- RIDGE ---
    num_pipe_ridge = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    pre_ridge = ColumnTransformer([
        ("num", num_pipe_ridge, num_cols),
        ("cat", make_ohe_sparse(), cat_cols)
    ], remainder="drop", sparse_threshold=1.0)

    ridge = MultiOutputRegressor(
        RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0], cv=3), n_jobs=None
    )
    pipe_ridge = Pipeline([("pre", pre_ridge), ("est", ridge)])
    pipe_ridge.fit(Xtr, ytr)
    pr = pipe_ridge.predict(Xte)

    ridge_res = {
        "model": "RidgeCV (no PCA, sparse OHE)",
        "rmse": float(np.sqrt(mean_squared_error(yte, pr))),
        "mae": float(mean_absolute_error(yte, pr)),
        "geo_km": geodesic_km_avg(yte, pr)
    }

    # --- MLP ---
    num_pipe_mlp = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    pre_mlp_sparse = ColumnTransformer([
        ("num", num_pipe_mlp, num_cols),
        ("cat", make_ohe_sparse(), cat_cols)
    ], remainder="drop", sparse_threshold=1.0)

    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=400,
        random_state=RND,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=12
    )

    mlp_pipeline = Pipeline([
        ("pre", pre_mlp_sparse),
        ("svd", TruncatedSVD(n_components=svd_components, random_state=RND)),
        ("sc2", StandardScaler()),
        ("est", MultiOutputRegressor(mlp))
    ])

    mlp_pipeline.fit(Xtr, ytr)
    pm = mlp_pipeline.predict(Xte)

    try:
        epoch_run = mlp_pipeline.named_steps["est"].estimator_.n_iter_
    except Exception:
        epoch_run = None

    svd_var = float(np.sum(mlp_pipeline.named_steps["svd"].explained_variance_ratio_))

    mlp_res = {
        "model": f"MLP (SVD-{svd_components}, early stop)",
        "rmse": float(np.sqrt(mean_squared_error(yte, pm))),
        "mae": float(mean_absolute_error(yte, pm)),
        "geo_km": geodesic_km_avg(yte, pm),
        "svd_var_kept": round(svd_var, 4),
        "epoch_run": int(epoch_run) if epoch_run is not None else None
    }

    del Xtr, Xte, ytr, yte
    gc.collect()

    return pd.DataFrame([ridge_res, mlp_res]), pipe_ridge, mlp_pipeline


# ==============================
# So sánh tỷ lệ train/test
# ==============================

def compare_train_test_ratios(dfs):
    _SAMPLE = 30000
    ratios = [0.2, 0.3, 0.4]

    results_ratios = {}
    for h in [6]:
        print(f"\n📊 [Quick Ratio Test] Horizon {h}h (sample={_SAMPLE} rows)")
        D = dfs[h].copy()
        if len(D) > _SAMPLE:
            D = D.sample(n=_SAMPLE, random_state=RND).reset_index(drop=True)

        rows = []
        for ts in ratios:
            print(f"  ▶️ Test_size={ts}  (train={1-ts:.1f})")
            res, _, _ = regression_optimized(D, test_size=ts, svd_components=64)
            res["test_size"] = ts
            rows.append(res)

        results_ratios[h] = pd.concat(rows, ignore_index=True)
        print(results_ratios[h])

    return results_ratios


def plot_ratio_comparison():
    """Vẽ biểu đồ so sánh RMSE / MAE / GeoKM theo tỷ lệ train-test."""
    data = pd.DataFrame({
        "model": [
            "RidgeCV (no PCA, sparse OHE)", "MLP (SVD-64, early stop)",
            "RidgeCV (no PCA, sparse OHE)", "MLP (SVD-64, early stop)",
            "RidgeCV (no PCA, sparse OHE)", "MLP (SVD-64, early stop)"
        ],
        "rmse": [0.307039, 3.098335, 0.304784, 3.127594, 0.309829, 3.381714],
        "mae": [0.198438, 1.944549, 0.200051, 2.023845, 0.203331, 2.204377],
        "geo_km": [32.699497, 356.523987, 32.913463, 362.509625, 33.422291, 391.054470],
        "test_size": [0.2, 0.2, 0.3, 0.3, 0.4, 0.4]
    })

    metrics = ["rmse", "mae", "geo_km"]
    titles = ["Sai số RMSE", "Sai số MAE", "Khoảng cách địa lý trung bình (km)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, m in enumerate(metrics):
        sns.barplot(data=data, x="test_size", y=m, hue="model", palette="Set2", ax=axes[i])
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Tỷ lệ test (test_size)")
        axes[i].set_ylabel(m.upper())
        axes[i].grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# ==============================
# So sánh Original vs PCA
# ==============================

def regression_compare_opt(D, use_pca=False, n_components=0.9, test_size=0.3):
    _SAMPLE = 30000
    D = D.sample(n=min(_SAMPLE, len(D)), random_state=RND).reset_index(drop=True)
    D = add_dynamic_features(D)

    cat_cols = D.select_dtypes("object").columns.tolist()
    num_cols = [c for c in D.select_dtypes("number").columns if c not in ("lat_tgt", "lon_tgt")]

    for c in cat_cols:
        D[c] = D[c].astype(str).fillna("__MISSING__")

    y = D[["lat_tgt", "lon_tgt"]].values
    groups = D["storm_id"].astype(str).values

    gss = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=RND)
    tr_idx, te_idx = next(gss.split(D, y, groups))
    Xtr, Xte, ytr, yte = D.iloc[tr_idx], D.iloc[te_idx], y[tr_idx], y[te_idx]

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    if use_pca:
        num_pipe.steps.append(("pca", PCA(n_components=n_components, random_state=RND)))

    preproc = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", make_ohe_sparse(), cat_cols)
    ], remainder="drop", sparse_threshold=1.0)

    ridge = MultiOutputRegressor(RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0], cv=3), n_jobs=None)
    pipe_ridge = Pipeline([("pre", preproc), ("est", ridge)])
    pipe_ridge.fit(Xtr, ytr)
    pr = pipe_ridge.predict(Xte)

    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, batch_size=512,
                       learning_rate_init=1e-3, max_iter=400,
                       early_stopping=True, validation_fraction=0.15,
                       n_iter_no_change=12, random_state=RND)
    pipe_mlp = Pipeline([("pre", preproc), ("est", MultiOutputRegressor(mlp))])
    pipe_mlp.fit(Xtr, ytr)
    pm = pipe_mlp.predict(Xte)

    return pd.DataFrame([
        {"model": f"RidgeCV {'+ PCA' if use_pca else '(original)'}",
         "rmse": np.sqrt(mean_squared_error(yte, pr)),
         "mae": mean_absolute_error(yte, pr),
         "geo_km": geodesic_km_avg(yte, pr)},
        {"model": f"MLP {'+ PCA' if use_pca else '(original)'}",
         "rmse": np.sqrt(mean_squared_error(yte, pm)),
         "mae": mean_absolute_error(yte, pm),
         "geo_km": geodesic_km_avg(yte, pm)}
    ])


# ==============================
# Kiểm tra Overfitting & Regularization
# ==============================

def sanitize_categoricals(df, cat_cols):
    df = df.copy()
    for c in cat_cols:
        df[c] = df[c].astype(str)
        df[c] = df[c].replace(["nan", "None", "NaN", "NaT", "NULL"], "__MISSING__")
        df[c] = df[c].fillna("__MISSING__")
    return df


def run_overfitting_check(dfs):
    H = 6
    _SAMPLE = 30000
    D = dfs[H].sample(_SAMPLE, random_state=42).copy()
    D = add_dynamic_features(D)

    for c in D.columns:
        if D[c].dtype == "object" or D[c].nunique() < 25:
            D[c] = D[c].astype(str).fillna("__MISSING__")

    y_all = D[["lat_tgt", "lon_tgt"]].values
    groups = D["storm_id"].astype(str).values
    gss = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)
    tr_idx, te_idx = next(gss.split(D, y_all, groups))
    Xtr, Xte = D.iloc[tr_idx], D.iloc[te_idx]
    ytr, yte = y_all[tr_idx], y_all[te_idx]

    cat_cols = Xtr.select_dtypes("object").columns.tolist()
    num_cols = [c for c in Xtr.select_dtypes("number").columns if c not in ("lat_tgt", "lon_tgt")]

    num_pipe_ridge = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    pre_ridge = ColumnTransformer([
        ("num", num_pipe_ridge, num_cols),
        ("cat", make_ohe_sparse(), cat_cols)
    ], sparse_threshold=1.0)

    ridge_model = MultiOutputRegressor(
        RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0], cv=3), n_jobs=None
    )
    ridge_pipe = Pipeline([("pre", pre_ridge), ("est", ridge_model)])
    ridge_pipe.fit(Xtr, ytr)

    ridge_train_rmse = np.sqrt(mean_squared_error(ytr, ridge_pipe.predict(Xtr)))
    ridge_test_rmse = np.sqrt(mean_squared_error(yte, ridge_pipe.predict(Xte)))

    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), solver="adam", alpha=1e-4,
                       batch_size=256, learning_rate_init=1e-3, max_iter=400,
                       random_state=42, early_stopping=True,
                       validation_fraction=0.15, n_iter_no_change=12)
    mlp_pipe = Pipeline([
        ("pre", pre_ridge),
        ("svd", TruncatedSVD(n_components=128, random_state=42)),
        ("sc2", StandardScaler()),
        ("est", MultiOutputRegressor(mlp))
    ])
    mlp_pipe.fit(Xtr, ytr)

    mlp_train_rmse = np.sqrt(mean_squared_error(ytr, mlp_pipe.predict(Xtr)))
    mlp_test_rmse = np.sqrt(mean_squared_error(yte, mlp_pipe.predict(Xte)))

    df_overfit = pd.DataFrame({
        "Model": ["RidgeCV (Optimized)", "MLP (Optimized)"],
        "Train RMSE": [ridge_train_rmse, mlp_train_rmse],
        "Test RMSE": [ridge_test_rmse, mlp_test_rmse],
        "Overfit Gap (ΔRMSE)": [
            ridge_test_rmse - ridge_train_rmse,
            mlp_test_rmse - mlp_train_rmse
        ]
    })
    print(df_overfit)

    x = np.arange(len(df_overfit))
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.15, df_overfit["Train RMSE"], width=0.3, label="Train")
    plt.bar(x + 0.15, df_overfit["Test RMSE"], width=0.3, label="Test")
    plt.xticks(x, df_overfit["Model"])
    plt.ylabel("RMSE")
    plt.title("So sánh Overfitting giữa RidgeCV và MLP")
    plt.legend()
    plt.show()

    gc.collect()
    return ridge_pipe, mlp_pipe, Xte, yte


def run_mlp_regularization(dfs):
    """MLP với regularization mạnh hơn để giảm overfit."""
    H = 6
    D = dfs[H].copy()
    D = add_dynamic_features(D)
    D = D.sample(30000, random_state=42).reset_index(drop=True)

    cat_cols = D.select_dtypes("object").columns.tolist()
    num_cols = [c for c in D.select_dtypes("number").columns if c not in ("lat_tgt", "lon_tgt")]

    if cat_cols:
        D = sanitize_categoricals(D, cat_cols)

    y_all = D[["lat_tgt", "lon_tgt"]].values
    groups = D["storm_id"].astype(str).values
    gss = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)
    tr_idx, te_idx = next(gss.split(D, y_all, groups))
    Xtr, Xte = D.iloc[tr_idx], D.iloc[te_idx]
    ytr, yte = y_all[tr_idx], y_all[te_idx]

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", make_ohe_sparse(), cat_cols)
    ], remainder="drop", sparse_threshold=1.0)

    mlp_reg = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        solver="adam",
        alpha=1e-3,
        batch_size=256,
        learning_rate_init=5e-4,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=15,
        random_state=42
    )

    mlp_pipe_reg = Pipeline([
        ("pre", pre),
        ("svd", TruncatedSVD(n_components=128, random_state=42)),
        ("sc2", StandardScaler()),
        ("est", MultiOutputRegressor(mlp_reg))
    ])

    mlp_pipe_reg.fit(Xtr, ytr)
    rmse_train = np.sqrt(mean_squared_error(ytr, mlp_pipe_reg.predict(Xtr)))
    rmse_test = np.sqrt(mean_squared_error(yte, mlp_pipe_reg.predict(Xte)))
    gap = rmse_test - rmse_train

    print("✅ Hiệu chỉnh Regularization cho MLP (Reduced Overfit)")
    print(f"Train RMSE: {rmse_train:.4f}")
    print(f"Test RMSE:  {rmse_test:.4f}")
    print(f"ΔRMSE (Overfit gap): {gap:.4f}")

    plt.figure(figsize=(6, 4))
    plt.bar(["Train", "Test"], [rmse_train, rmse_test], color=["skyblue", "salmon"])
    plt.title("Hiệu quả sau khi Regularization MLP (Giảm Overfit)")
    plt.ylabel("RMSE")
    plt.show()

    gc.collect()
    return mlp_pipe_reg, Xte, yte


# ==============================
# Residual Analysis
# ==============================

def sanitize_for_predict(df, model):
    df = df.copy()
    cat_cols_in_model = model.named_steps["pre"].transformers_[1][2]
    for c in cat_cols_in_model:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("__MISSING__")
    return df


def plot_residuals(y_true, y_pred, name):
    residuals = y_true - y_pred
    res_mag = np.sqrt((residuals ** 2).sum(axis=1))
    rmse = np.sqrt(np.mean(res_mag ** 2))
    std = np.std(res_mag)

    print(f"\n📊 {name}: RMSE={rmse:.4f}, Std={std:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(res_mag, bins=40, kde=True, color="salmon")
    plt.title(f"Phân phối phần dư — {name}")
    plt.xlabel("Độ lớn phần dư")
    plt.ylabel("Tần suất")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_pred[:, 0], y=res_mag, alpha=0.4, s=10, color="royalblue")
    plt.title(f"Phần dư theo vĩ độ dự đoán — {name}")
    plt.xlabel("Vĩ độ dự đoán (lat_pred)")
    plt.ylabel("Độ lớn phần dư")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _, dfs = run_preprocessing()

    print("\n=== CHẠY CHO TỪNG HORIZON ===")
    results_opt = {}
    for h in HORIZONS:
        print(f"\n[Optimized Regression] Horizon {h}h (sample={SAMPLE_SIZE})")
        D = dfs[h].copy()
        results_opt[h], _, _ = regression_optimized(D, test_size=0.3, svd_components=128)
        print(results_opt[h])

    plot_ratio_comparison()

    H = 6
    D = dfs[H].copy()
    res_orig = regression_compare_opt(D, use_pca=False)
    res_pca = regression_compare_opt(D, use_pca=True)
    compare_opt = pd.concat([res_orig, res_pca], ignore_index=True)
    print("\n=== Original vs PCA ===")
    print(compare_opt)

    ridge_pipe, mlp_pipe, Xte, yte = run_overfitting_check(dfs)
    mlp_pipe_reg, Xte_reg, yte_reg = run_mlp_regularization(dfs)

    # Residual analysis
    for name, (model, X, y) in {
        "RidgeCV": (ridge_pipe, Xte, yte),
        "MLP (Regularized)": (mlp_pipe_reg, Xte_reg, yte_reg)
    }.items():
        try:
            Xsafe = sanitize_for_predict(X, model)
        except Exception:
            Xsafe = X
        plot_residuals(y, model.predict(Xsafe), name)
