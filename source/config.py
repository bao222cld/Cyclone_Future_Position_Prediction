"""
config.py — Cấu hình toàn cục cho dự án dự báo quỹ đạo bão
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np

RND = 42
np.random.seed(RND)

DATA_PATH = "/content/typhoon_data_1970.csv"  # thay bằng đường dẫn đúng
HORIZONS = [6, 8, 12]  # giờ dự báo phía trước

# Tùy chọn hiển thị/đồ thị nhẹ
FAST_PLOTS = True

SAMPLE_SIZE = None  # None = dùng full dữ liệu; ví dụ 30000 để test nhanh
