
"""Tạo CSV demo gồm 60 sinh viên thuộc tập Test UCI 697."""

from pathlib import Path
import re
import unicodedata

import pandas as pd
from ucimlrepo import fetch_ucirepo


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PREDICTIONS = ROOT / "ket_qua_v3" / "test_predictions.csv"
OUTPUT = DATA_DIR / "test_students.csv"

N_ROWS = 60
SEED = 42


def clean_column_name(value):
    return str(value).replace("\ufeff", "").strip()


def norm(value):
    s = unicodedata.normalize("NFKD", clean_column_name(value))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def is_semester2(column):
    name = norm(column)
    return (
        "2ndsem" in name
        or "secondsem" in name
        or "semester2" in name
    )


def main():
    if not PREDICTIONS.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {PREDICTIONS}. Hãy chạy run_v3.py trước."
        )

    predictions = pd.read_csv(PREDICTIONS)
    predictions.columns = [clean_column_name(c) for c in predictions.columns]

    if "index" not in predictions.columns:
        raise ValueError(
            "test_predictions.csv không có cột 'index'. "
            "Cần kiểm tra cách run_v3.py lưu chỉ số tập Test."
        )

    test_indices = (
        pd.to_numeric(predictions["index"], errors="raise")
        .astype(int)
        .tolist()
    )

    if not test_indices:
        raise ValueError("Danh sách mẫu Test đang trống.")

    # Chọn 60 mẫu thuộc tập Test
    if len(test_indices) > N_ROWS:
        selected_indices = (
            pd.Series(test_indices)
            .sample(n=N_ROWS, random_state=SEED)
            .tolist()
        )
    else:
        selected_indices = test_indices

    print("Đang tải dữ liệu UCI 697...")
    dataset = fetch_ucirepo(id=697)

    X = dataset.data.features.copy()
    X.columns = [clean_column_name(c) for c in X.columns]
    X = X.reset_index(drop=True)

    # Chỉ loại đặc trưng học kỳ 2
    semester2_columns = [c for c in X.columns if is_semester2(c)]
    X = X.drop(columns=semester2_columns, errors="ignore")

    # Tuyệt đối không đưa nhãn Target vào đầu vào
    if "Target" in X.columns:
        X = X.drop(columns=["Target"])

    if min(selected_indices) < 0 or max(selected_indices) >= len(X):
        raise IndexError(
            "Chỉ số mẫu trong test_predictions.csv không khớp "
            "với thứ tự dữ liệu UCI 697."
        )

    sample = X.iloc[selected_indices].copy()

    # Chuẩn hóa tên cột Marital status nếu có khác biệt về BOM/khoảng trắng
    marital_candidates = [
        c for c in sample.columns if norm(c) == norm("Marital status")
    ]
    if marital_candidates:
        sample = sample.rename(
            columns={marital_candidates[0]: "Marital status"}
        )
    else:
        raise ValueError(
            "Không tìm thấy cột Marital status trong dữ liệu nguồn. "
            f"Các cột hiện có: {sample.columns.tolist()}"
        )

    # Thêm mã định danh demo
    sample.insert(
        0,
        "Student_ID",
        [f"TEST_{i:03d}" for i in range(1, len(sample) + 1)],
    )

    # App hiện báo thiếu tên có BOM; tạo tên cột tương ứng trong CSV.
    sample.columns = [
        "\ufeffMarital status" if c == "Marital status" else c
        for c in sample.columns
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Dùng UTF-8 không BOM để tránh tự thêm BOM trước cột Student_ID.
    sample.to_csv(OUTPUT, index=False, encoding="utf-8")

    print("\nĐÃ TẠO CSV DEMO")
    print(f"File: {OUTPUT}")
    print(f"Số sinh viên: {len(sample)}")
    print(f"Số cột gồm Student_ID: {len(sample.columns)}")
    print(f"Số cột học kỳ 2 đã loại: {len(semester2_columns)}")
    print("Không chứa Target.")
    print("Tên cột Marital status đã được xử lý theo lỗi app.")
    print("\nDanh sách cột:")
    print([repr(c) for c in sample.columns])


if __name__ == "__main__":
    main()
