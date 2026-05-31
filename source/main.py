"""
main.py — Điểm chạy toàn bộ pipeline
Chạy tuần tự: Preprocessing → EDA → Clustering → Regression → Classification
"""

from config import DATA_PATH, HORIZONS

print("=" * 60)
print("  TYPHOON TRAJECTORY PREDICTION PIPELINE")
print("=" * 60)

# ------ 1. Preprocessing ------
print("\n[1/5] TIỀN XỬ LÝ DỮ LIỆU...")
from preprocessing import run_preprocessing
df, dfs = run_preprocessing()

# ------ 2. EDA ------
print("\n[2/5] PHÂN TÍCH VÀ TRỰC QUAN HÓA...")
from eda import (
    plot_feature_correlation, plot_pca_visualization,
    evaluate_pca_lda, plot_lda_distribution, describe_normalized
)
plot_feature_correlation(dfs)
plot_pca_visualization(dfs)
Xn, labels = evaluate_pca_lda(dfs)
plot_lda_distribution(Xn, labels)
describe_normalized(dfs)

# ------ 3. Clustering ------
print("\n[3/5] PHÂN CỤM DỮ LIỆU...")
from clustering import build_output_labels, run_clustering
D_outlabels = build_output_labels(dfs)
D_outlabels = run_clustering(D_outlabels)

# ------ 4. Regression ------
print("\n[4/5] PHÂN TÍCH HỒI QUY...")
from regression import (
    regression_optimized, plot_ratio_comparison,
    regression_compare_opt, run_overfitting_check,
    run_mlp_regularization, sanitize_for_predict, plot_residuals
)
from config import SAMPLE_SIZE

results_opt = {}
for h in HORIZONS:
    print(f"\n[Optimized Regression] Horizon {h}h")
    D = dfs[h].copy()
    results_opt[h], _, _ = regression_optimized(D, test_size=0.3, svd_components=128)
    print(results_opt[h])

plot_ratio_comparison()

H = 6
D = dfs[H].copy()
res_orig = regression_compare_opt(D, use_pca=False)
res_pca  = regression_compare_opt(D, use_pca=True)
import pandas as pd
print("\n=== Original vs PCA ===")
print(pd.concat([res_orig, res_pca], ignore_index=True))

ridge_pipe, mlp_pipe, Xte_reg, yte_reg = run_overfitting_check(dfs)
mlp_pipe_reg, Xte_reg2, yte_reg2 = run_mlp_regularization(dfs)

for name, (model, X, y) in {
    "RidgeCV": (ridge_pipe, Xte_reg, yte_reg),
    "MLP (Regularized)": (mlp_pipe_reg, Xte_reg2, yte_reg2)
}.items():
    try:
        Xsafe = sanitize_for_predict(X, model)
    except Exception:
        Xsafe = X
    plot_residuals(y, model.predict(Xsafe), name)

# ------ 5. Classification ------
print("\n[5/5] PHÂN LOẠI...")
from classification import (
    prepare_classification_data, run_classification_experiments,
    print_summary, run_pca_classification, plot_pca_comparison,
    run_xgboost, plot_model_comparison, data_leakage_check
)

X, y, groups = prepare_classification_data(D_outlabels)
data_leakage_check(X, y)
nb, rf, Xtr, Xte, ytr, yte, le = run_classification_experiments(X, y, groups)
print_summary()
run_pca_classification(Xtr, Xte, ytr, yte)
plot_pca_comparison()
xgb = run_xgboost(Xtr, Xte, ytr, yte, le)
plot_model_comparison(nb, rf, xgb, Xtr, Xte, ytr, yte)

print("\n✅ Pipeline hoàn tất!")
