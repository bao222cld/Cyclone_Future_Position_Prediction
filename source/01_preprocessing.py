"""
01_preprocessing.py — Tiền xử lý dữ liệu
  - Đọc dữ liệu, mô tả cấu trúc
  - Xử lý lỗi, thiếu, định dạng sai
  - Utilities (Haversine, GroupShuffleSplit wrapper)
  - Feature Engineering (leak-free)
  - Build Horizon Datasets (6h, 8h, 12h)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from config import DATA_PATH, HORIZONS, RND, FAST_PLOTS

plt.rcParams.update({"figure.figsize": (6, 5), "axes.grid": True})


# ==============================
# Mô tả cấu trúc và thống kê sơ bộ về dữ liệu
# ==============================

def describe_raw(path):
    df = pd.read_csv(path)

    print("📊 TỔNG QUAN DỮ LIỆU")
    print(f"- Số bản ghi (rows): {df.shape[0]:,}")
    print(f"- Số trường (columns): {df.shape[1]}")
    print("\n🧩 DANH SÁCH CÁC TRƯỜNG:")
    for col in df.columns:
        print(f"  • {col}")

    print("\n📅 KIỂU DỮ LIỆU & GIÁ TRỊ KHÔNG RỖNG:")
    df.info()

    print("\n📈 THỐNG KÊ SƠ BỘ CÁC CỘT SỐ:")
    print(df.describe().T[['count', 'mean', 'std', 'min', 'max']].round(2))

    print("\n🔍 MỘT VÀI DÒNG DỮ LIỆU ĐẦU TIÊN:")
    print(df.head())

    # Kiểm tra giá trị thiếu
    missing = df.isna().sum()
    if missing.sum() > 0:
        print("\n⚠️ CÁC TRƯỜNG CÒN THIẾU GIÁ TRỊ:")
        print(missing[missing > 0].sort_values(ascending=False))
    else:
        print("\n✅ KHÔNG CÓ GIÁ TRỊ THIẾU TRONG DỮ LIỆU.")

    return df


# ==============================
# Load & Basic Cleaning  (FIX: normalize time tz → naive)
# ==============================

def memory_efficient_read_csv(path):
    """Đọc và làm sạch cơ bản file CSV."""
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", low_memory=True)

    # Chuẩn hóa tên cột
    df.columns = df.columns.str.lower().str.strip()

    # Chuẩn hóa datetime
    if "iso_time" in df.columns:
        df["iso_time"] = pd.to_datetime(df["iso_time"], errors="coerce", utc=True)
        df = df.rename(columns={"iso_time": "time"})
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

    # Chuẩn hóa về cùng 1 time zone UTC
    if "time" in df.columns:
        if hasattr(df["time"].dt, "tz"):
            try:
                df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)
            except Exception:
                df["time"] = df["time"].dt.tz_localize(None)

    # Chuẩn hóa tên các cột chính
    df = df.rename(columns={
        "sid": "storm_id",
        "wmo_wind": "wind",
        "wmo_pres": "pres",
    })

    # Lọc những hàng có tọa độ hợp lệ
    df = df.dropna(subset=["lat", "lon"])

    # Sắp xếp theo storm_id, time
    key_time = "time" if "time" in df.columns else None
    if key_time:
        df = df.sort_values(["storm_id", key_time])
    else:
        df = df.sort_values(["storm_id"])

    df.reset_index(drop=True, inplace=True)
    return df


# ==============================
# Utilities (vectorized)
# ==============================
# CHÚ Ý : LAT: VĨ ĐỘ, LON: KINH ĐỘ

def haversine_vec(lat1, lon1, lat2, lon2):
    """Tính khoảng cách địa lý giữa 2 điểm (km), vectorized."""
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R = 6371.0
    return R * c


def group_split_indices(X, y, groups, test_size):
    """Chia dữ liệu train/test mà vẫn giữ toàn vẹn từng cơn bão (storm_id)."""
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=RND)
    return next(splitter.split(X, y, groups))


# ==============================
# 3) Feature Engineering (leak-free)  (FIX: robust datetime dtype check)
# ==============================

def add_time_features(d):
    """Thêm đặc trưng thời gian: month, hour, dayofyear."""
    d = d.copy()
    if "time" in d.columns and pd.api.types.is_datetime64_any_dtype(d["time"]):
        try:
            if hasattr(d["time"].dt, "tz") and d["time"].dt.tz is not None:
                d["time"] = d["time"].dt.tz_localize(None)
        except Exception:
            pass
        d["month"] = d["time"].dt.month.astype("Int16")
        d["hour"] = d["time"].dt.hour.astype("Int16")
        d["dayofyear"] = d["time"].dt.dayofyear.astype("Int16")
    else:
        d["month"] = pd.Series(pd.NA, index=d.index, dtype="Int16")
        d["hour"] = pd.Series(pd.NA, index=d.index, dtype="Int16")
        d["dayofyear"] = pd.Series(pd.NA, index=d.index, dtype="Int16")
    return d


def add_lag_kinematics(d, max_lag_steps=2):
    """Thêm đặc trưng động học từ dữ liệu quá khứ (lag/rolling)."""
    d = d.copy()
    d = d.sort_values(["storm_id", "time"]).reset_index(drop=True)

    for L in range(1, max_lag_steps + 1):
        d[f"lat_lag{L}"] = d.groupby("storm_id")["lat"].shift(L)
        d[f"lon_lag{L}"] = d.groupby("storm_id")["lon"].shift(L)
        d[f"dlat_lag{L}"] = d["lat"] - d[f"lat_lag{L}"]
        d[f"dlon_lag{L}"] = d["lon"] - d[f"lon_lag{L}"]

    lat_lag1 = d["lat_lag1"].to_numpy()
    lon_lag1 = d["lon_lag1"].to_numpy()
    lat_cur = d["lat"].to_numpy()
    lon_cur = d["lon"].to_numpy()
    step_km = np.full(len(d), np.nan, dtype=float)
    mask = (~pd.isna(lat_lag1)) & (~pd.isna(lon_lag1))
    step_km[mask] = haversine_vec(lat_lag1[mask], lon_lag1[mask], lat_cur[mask], lon_cur[mask])
    d["step_km_lag1"] = step_km

    d["step_km_roll3"] = (
        d.groupby("storm_id")["step_km_lag1"]
         .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )
    return d


def add_dynamic_features(D):
    """Thêm đặc trưng động học mở rộng: tốc độ, hướng di chuyển, gia tốc."""
    D = D.copy()
    D["speed_kmh"] = D["step_km_lag1"] / 3.0
    D["bearing"] = np.degrees(np.arctan2(D["dlon_lag1"], D["dlat_lag1"]))
    D["bearing"] = D["bearing"].fillna(0)
    D["dlat_acc"] = D["dlat_lag1"] - D["dlat_lag2"]
    D["dlon_acc"] = D["dlon_lag1"] - D["dlon_lag2"]
    D["dist2eq"] = np.abs(D["lat"])
    return D


def build_horizon(df_in, horizon_hours=6, base_step_hours=3):
    """Tạo cột target (lat_tgt, lon_tgt) cho mốc horizon_hours giờ tương lai."""
    steps = max(1, int(round(horizon_hours / base_step_hours)))
    d = df_in.copy()
    d = d.sort_values(["storm_id", "time"]).reset_index(drop=True)
    d["lat_tgt"] = d.groupby("storm_id")["lat"].shift(-steps)
    d["lon_tgt"] = d.groupby("storm_id")["lon"].shift(-steps)
    d = d.dropna(subset=["lat_tgt", "lon_tgt"]).reset_index(drop=True)
    return d


# ==============================
# Pipeline chính
# ==============================

def run_preprocessing():
    print("[1] Loading…")
    df = memory_efficient_read_csv(DATA_PATH)
    print(f"Shape: {df.shape} | time dtype: {df['time'].dtype if 'time' in df.columns else 'N/A'}")

    print("[3] Feature Engineering (leak-free)…")
    df_feat = add_time_features(df)
    df_feat = add_lag_kinematics(df_feat, max_lag_steps=2)
    print("Features added. Columns:", len(df_feat.columns))

    print("[4] Build Horizon Datasets…")
    dfs = {}
    for h in HORIZONS:
        D = build_horizon(df_feat, horizon_hours=h, base_step_hours=3)
        dfs[h] = D
        print(f"  Horizon {h}h → {len(D):,} rows")

    print("[5] Adding dynamic features…")
    dfs = {h: add_dynamic_features(df) for h, df in dfs.items()}
    print("  Dynamic features added: speed_kmh, bearing, dlat_acc, dlon_acc, dist2eq")

    return df, dfs


if __name__ == "__main__":
    df, dfs = run_preprocessing()
