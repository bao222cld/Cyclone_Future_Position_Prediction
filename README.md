#  Typhoon Trajectory Prediction

Dự án phân tích và dự báo quỹ đạo bão từ dữ liệu lịch sử (IBTrACS từ năm 1970), bao gồm tiền xử lý dữ liệu, trực quan hóa, phân cụm, hồi quy và phân loại.

>  **Notebook đầy đủ với output:** [Xem trên Google Colab]([PASTE_COLAB_LINK_HERE](https://colab.research.google.com/drive/1eHNeK8yVLN-5Mm47Wif7hV__bENyAZaE?authuser=1))

---

##  Cấu trúc file

```
├── config.py               # Cấu hình toàn cục (DATA_PATH, HORIZONS, RND...)
├── 01_preprocessing.py     # Tiền xử lý, feature engineering, build horizon datasets
├── 02_eda.py               # Phân tích và trực quan hóa (tương quan, PCA, LDA)
├── 03_clustering.py        # Phân cụm K-Means & GMM, tạo nhãn output
├── 04_regression.py        # Hồi quy RidgeCV + MLP, so sánh, overfitting check
├── 05_classification.py    # Phân loại NB / RF / XGBoost, PCA + classification
├── main.py                 # Chạy toàn bộ pipeline tuần tự
└── README.md
```

---

##  Dữ liệu

File dữ liệu: `typhoon_data_1970.csv` (IBTrACS — không được commit vào repo do kích thước lớn).

Tải dữ liệu tại: https://www.ncei.noaa.gov/products/international-best-track-archive

Sau khi tải, cập nhật đường dẫn trong `config.py`:
```python
DATA_PATH = "/path/to/typhoon_data_1970.csv"
```

---

##  Cài đặt

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

---

##  Chạy

### Chạy toàn bộ pipeline:
```bash
python main.py
```

### Chạy từng bước:
```bash
python 01_preprocessing.py   # Tiền xử lý
python 02_eda.py              # Phân tích EDA
python 03_clustering.py       # Phân cụm
python 04_regression.py       # Hồi quy
python 05_classification.py   # Phân loại
```

---

##  Nội dung chính

### 1. Tiền xử lý (`01_preprocessing.py`)
- Đọc và chuẩn hóa dữ liệu thô (timezone, tên cột, giá trị thiếu)
- Feature engineering: lag kinematics (lat/lon lag 1–2, step_km, rolling mean)
- Dynamic features: speed_kmh, bearing, acceleration, dist2equator
- Build horizon datasets: 6h, 8h, 12h (dự báo phía trước)

### 2. EDA (`02_eda.py`)
- Tương quan Pearson của đặc trưng với lat_tgt / lon_tgt
- PCA visualization (6 thành phần chính)
- Đánh giá PCA & LDA theo số chiều
- Thống kê dữ liệu sau chuẩn hóa (StandardScaler)

### 3. Phân cụm (`03_clustering.py`)
- K-Means và GMM (k=4) trên đặc trưng đầu vào
- Đánh giá: Silhouette score, Davies–Bouldin index
- Trực quan hóa trên không gian PCA(2)
- Tạo nhãn sai lệch (deviation) từ baseline tuyến tính

### 4. Hồi quy (`04_regression.py`)
- **RidgeCV**: sparse OHE, không PCA — nhanh và ổn định
- **MLP**: TruncatedSVD(128) + early stopping — học quan hệ phi tuyến
- So sánh 3 tỷ lệ train/test: 4:1, 7:3, 6:4
- So sánh Original vs PCA
- Kiểm tra và giảm overfitting (regularization)
- Residual analysis

### 5. Phân loại (`05_classification.py`)
- Nhãn: 4 lớp theo phân vị độ lớn sai lệch (Small → Large Error)
- Mô hình: Naive Bayes, Random Forest, XGBoost
- Đánh giá: F1-macro, confusion matrix, per-class accuracy
- So sánh với/không có PCA
- Data leakage sanity check

---

##  Kết quả tóm tắt

| Bài toán | Mô hình tốt nhất | Chỉ số |
|---|---|---|
| Hồi quy (6h) | RidgeCV | GeoKM ≈ 32.7 km |
| Phân loại | Random Forest | F1-macro ≈ 0.69–0.71 |

---

## 👤 Tác giả

Nguyễn Lê Ngọc Bảo — Nhóm 8
