# Bài cuối kỳ Nhận dạng mẫu – Nhóm 12: mã nguồn thực nghiệm

Đề tài: Dự đoán và cảnh báo sớm nguy cơ bỏ học của sinh viên, bộ dữ liệu UCI 697
(Predict Students' Dropout and Academic Success, https://archive.ics.uci.edu/dataset/697, giấy phép CC BY 4.0).

## Cài đặt

Python 3.12.

```bash
pip install numpy pandas scipy scikit-learn matplotlib catboost lightgbm ucimlrepo
```

## Giao thức

1. Bỏ 6 cột học kỳ 2. Chia Train/Test 80:20 phân tầng theo ba lớp (random_state = 42): Train 3.539, Test 885
   sinh viên (284 Dropout). Test chỉ chạy 1 lần duy nhất
2. Stratified 5-fold CV trên Train, mọi bước tiền xử lý học lại trong từng fold, lưu dự đoán out-of-fold (OOF).
3. Mỗi cấu hình chọn ngưỡng τ* cho F2 (lớp Dropout) lớn nhất trên OOF của Train, quét 0,05–0,95, rồi khóa.
4. Học lại trên toàn bộ Train, đánh giá Test một lần tại τ*.

So sánh với Base (Logistic Regression, Original) bằng kiểm định t hiệu chỉnh Nadeau–Bengio trên 5 fold và bootstrap
ghép cặp 1.000 lần trên Test. Không tinh chỉnh siêu tham số.

## Chạy

```bash
python run_v3.py --data data/data.csv    # đọc file data.csv
python run_v3.py                         # hoặc tải dữ liệu qua ucimlrepo
```

## Đầu ra trong `ket_qua_v3/`

| File | Nội dung |
|---|---|
| `cv_fold_results.csv`, `cv_summary.csv` | Chỉ số từng fold và trung bình ± độ lệch chuẩn trên CV của Train |
| `thresholds.csv` | Ngưỡng τ* của từng cấu hình, F2 OOF tại τ* và tại 0,5 |
| `test_metrics.csv` | Kết quả trên Test tại τ* (F2, Recall, Precision, Specificity, AP, ROC-AUC, Brier, log loss, …) |
| `comparison_vs_base.csv`, `ablation_fe.csv` | Chênh lệch so với Base và tác động của FE, kèm p-value và khoảng tin cậy |
| `odds_ratio.csv` | Hệ số và odds ratio của Logistic Regression nhị phân Dropout so với phần còn lại |
| `error_groups_train_oof.csv`, `summary.json` | Phân tích lỗi: bỏ sót theo nhóm, ma trận lỗi, McNemar |
| `oof_train_predictions.csv`, `test_predictions.csv` | p(Dropout) của từng sinh viên (cột `index` là số thứ tự dòng trong data.csv) |

### Demo

Cài đặt thêm thư viện cần thiết:

```bash
pip install streamlit
```

**Bước 1 – Tạo dữ liệu CSV mẫu từ tập Test:**

```bash
python 1_generate_test_csv.py
```

Dữ liệu mẫu được lưu tại `data/test_students.csv`.

**Bước 2 – Huấn luyện và lưu mô hình:**

```bash
python 2_train_save_models.py
```

Script này huấn luyện và lưu các mô hình cùng các thành phần cần thiết cho demo.

**Bước 3 – Khởi chạy giao diện:**

```bash
streamlit run 3_app.py
```

Sau khi khởi chạy, mở địa chỉ Local URL được hiển thị trong terminal, thường là `http://localhost:8501`.

**Bước 4 – Thực hiện dự đoán:**

1. Tải lên tệp CSV chứa dữ liệu của nhiều sinh viên.
2. Kiểm tra dữ liệu đầu vào và các cột đặc trưng được yêu cầu.
3. Chạy dự đoán để xem xác suất nguy cơ bỏ học và kết quả cảnh báo.
4. Tải xuống tệp CSV kết quả nếu cần.

### Lưu ý

- Demo sử dụng kịch bản dự đoán sau học kỳ 1, không sử dụng các đặc trưng học kỳ 2.
- Mô hình và ngưỡng cảnh báo phải nhất quán với cấu hình thực nghiệm. Nếu sử dụng ngưỡng lấy từ `thresholds.csv`, cần đối chiếu đúng cấu hình mô hình tương ứng.
- Chạy bước 1 và bước 2 trước bước 3 để bảo đảm dữ liệu mẫu và các tệp mô hình đã được tạo đầy đủ.

