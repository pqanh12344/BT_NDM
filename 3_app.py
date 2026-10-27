"""Streamlit demo: chi nap model da luu, khong huan luyen lai."""
from pathlib import Path
import re
import unicodedata
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
CLASSES = ["Dropout", "Enrolled", "Graduate"]

st.set_page_config(page_title="Cảnh báo sớm nguy cơ bỏ học", page_icon="🎓", layout="wide")


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def is_semester2(c):
    n = norm(c)
    return "2ndsem" in n or "secondsem" in n or "semester2" in n


def find_col(df, wanted):
    lookup = {norm(c): c for c in df.columns}
    for w in wanted:
        if norm(w) in lookup:
            return lookup[norm(w)]
    return None


def add_fe(X):
    Z = X.copy()
    approved_c = find_col(Z, ["Curricular units 1st sem (approved)"])
    eval_c = find_col(Z, ["Curricular units 1st sem (evaluations)"])
    enrolled_c = find_col(Z, ["Curricular units 1st sem (enrolled)"])
    if not all([approved_c, eval_c, enrolled_c]):
        raise ValueError("CSV thiếu một trong ba cột học kỳ 1: approved, evaluations, enrolled.")
    approved = pd.to_numeric(Z[approved_c], errors="coerce")
    evaluations = pd.to_numeric(Z[eval_c], errors="coerce")
    enrolled = pd.to_numeric(Z[enrolled_c], errors="coerce")
    Z["FE_approval_rate"] = approved.div(evaluations).where(evaluations.ne(0), 0.0)
    Z["FE_evaluation_rate"] = evaluations.div(enrolled).where(enrolled.ne(0), 0.0)
    Z["FE_completion_rate"] = approved.div(enrolled).where(enrolled.ne(0), 0.0)
    Z["FE_no_eval_1"] = (evaluations == 0).astype(int)
    Z["FE_zero_enrolled_1"] = (enrolled == 0).astype(int)
    return Z.replace([np.inf, -np.inf], np.nan)


def load_artifacts():
    lr_path = MODEL_DIR / "logistic_regression.joblib"
    cb_path = MODEL_DIR / "catboost_fe.joblib"
    if not lr_path.exists() or not cb_path.exists():
        return None, None, "Chưa có model. Hãy chạy `python 2_train_save_models.py` trước."
    try:
        return joblib.load(lr_path), joblib.load(cb_path), None
    except Exception as e:
        return None, None, f"Không thể đọc model: {e}"


def prepare_input(df, expected_cols):
    # Student_ID la cot tuy chon chi de hien thi, khong dua vao mo hinh.
    X = df.drop(columns=[c for c in df.columns if norm(c) in {"target", "studentid", "id"}], errors="ignore").copy()
    X = X.loc[:, [c for c in X.columns if not is_semester2(c)]]
    missing = [c for c in expected_cols if c not in X.columns]
    extra = [c for c in X.columns if c not in expected_cols]
    if missing:
        raise ValueError(
            f"CSV thiếu {len(missing)} cột cần thiết so với dữ liệu train. Ví dụ: {missing[:8]}. "
            "Hãy dùng CSV mẫu tạo từ file 1 hoặc cung cấp đủ cột UCI 697 tương ứng."
        )
    # Bỏ cột thừa để bảo đảm cùng schema và thứ tự như lúc train.
    return X[expected_cols].copy(), extra


def predict_lr(bundle, X):
    model = bundle["model"]
    p = model.predict_proba(X)
    model_classes = list(model.classes_)
    aligned = np.zeros((len(X), len(CLASSES)), dtype=float)
    for j, label in enumerate(model_classes):
        aligned[:, CLASSES.index(label)] = p[:, j]
    return aligned


def predict_cb(bundle, X):
    Z = add_fe(X)
    expected = bundle["feature_columns"]
    missing = [c for c in expected if c not in Z.columns]
    if missing:
        raise ValueError(f"Thiếu đặc trưng CatBoost: {missing[:8]}")
    Z = Z[expected].copy()
    for c in bundle["categorical_columns"]:
        Z[c] = Z[c].where(Z[c].notna(), "__MISSING__").astype(str)
    model = bundle["model"]
    p = model.predict_proba(Z)
    model_classes = list(model.classes_)
    aligned = np.zeros((len(Z), len(CLASSES)), dtype=float)
    for j, label in enumerate(model_classes):
        aligned[:, CLASSES.index(label)] = p[:, j]
    return aligned


st.title("🎓 Dự đoán và cảnh báo sớm nguy cơ bỏ học")
st.caption("UCI 697 · Kịch bản T1: sau học kỳ 1 · Không sử dụng đặc trưng học kỳ 2")
st.info("Kết quả là tín hiệu sàng lọc hỗ trợ, không khẳng định sinh viên chắc chắn sẽ bỏ học.")

lr_bundle, cb_bundle, load_error = load_artifacts()
if load_error:
    st.error(load_error)
    st.stop()

with st.sidebar:
    st.header("Cấu hình")
    choice = st.radio("Mô hình", ["So sánh cả hai mô hình", "Logistic Regression", "CatBoost + FE"])
    st.write(f"Ngưỡng LR: **{float(lr_bundle['threshold']):.3f}**")
    st.write(f"Ngưỡng CatBoost + FE: **{float(cb_bundle['threshold']):.3f}**")
    st.caption("Ngưỡng được nạp từ artifacts khi train; demo không tự tối ưu ngưỡng.")

uploaded = st.file_uploader("Tải CSV chứa nhiều sinh viên", type=["csv"])
if uploaded is None:
    st.markdown("#### Bắt đầu nhanh")
    st.write("Chạy file `1_generate_test_csv.py` để tạo `data/test_students.csv`, sau đó tải file đó lên đây.")
    st.stop()

try:
    try:
        input_df = pd.read_csv(uploaded, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        uploaded.seek(0)
        input_df = pd.read_csv(uploaded, sep=None, engine="python", encoding="latin1")
    input_df.columns = input_df.columns.astype(str).str.strip()
    if input_df.empty:
        st.error("CSV không có bản ghi.")
        st.stop()

    expected_cols = lr_bundle["raw_columns"]
    X, extra_cols = prepare_input(input_df, expected_cols)
    if extra_cols:
        st.warning(f"Đã bỏ qua {len(extra_cols)} cột không dùng trong mô hình, ví dụ: {extra_cols[:5]}.")
    if any(is_semester2(c) for c in input_df.columns):
        st.info("Các cột học kỳ 2 đã được loại khỏi đầu vào trước khi dự đoán.")

    st.write(f"Số sinh viên trong file: **{len(X):,}**")
    st.dataframe(input_df.head(10), use_container_width=True)

    if st.button("Dự đoán nguy cơ", type="primary", use_container_width=True):
        results = []
        if choice in ["So sánh cả hai mô hình", "Logistic Regression"]:
            p_lr = predict_lr(lr_bundle, X)
            out_lr = pd.DataFrame({
                "Model": "Logistic Regression",
                "Prob_Dropout": p_lr[:, CLASSES.index("Dropout")],
                "Prob_Enrolled": p_lr[:, CLASSES.index("Enrolled")],
                "Prob_Graduate": p_lr[:, CLASSES.index("Graduate")],
                "Predicted_Class": [CLASSES[i] for i in np.argmax(p_lr, axis=1)],
                "Dropout_Risk": np.where(p_lr[:, CLASSES.index("Dropout")] >= float(lr_bundle["threshold"]), "Có nguy cơ - cần xem xét hỗ trợ", "Chưa vượt ngưỡng cảnh báo"),
                "Threshold_Dropout": float(lr_bundle["threshold"])
            })
            results.append(out_lr)
        if choice in ["So sánh cả hai mô hình", "CatBoost + FE"]:
            p_cb = predict_cb(cb_bundle, X)
            out_cb = pd.DataFrame({
                "Model": "CatBoost + FE",
                "Prob_Dropout": p_cb[:, CLASSES.index("Dropout")],
                "Prob_Enrolled": p_cb[:, CLASSES.index("Enrolled")],
                "Prob_Graduate": p_cb[:, CLASSES.index("Graduate")],
                "Predicted_Class": [CLASSES[i] for i in np.argmax(p_cb, axis=1)],
                "Dropout_Risk": np.where(p_cb[:, CLASSES.index("Dropout")] >= float(cb_bundle["threshold"]), "Có nguy cơ - cần xem xét hỗ trợ", "Chưa vượt ngưỡng cảnh báo"),
                "Threshold_Dropout": float(cb_bundle["threshold"])
            })
            results.append(out_cb)

        predictions = pd.concat(results, ignore_index=True)
        id_col = next((c for c in input_df.columns if norm(c) in {"studentid", "id"}), None)
        if id_col:
            ids = input_df[id_col].astype(str).tolist()
            predictions.insert(0, "Student_ID", ids * len(results))
        else:
            predictions.insert(0, "Student_ID", [f"ROW_{i+1:04d}" for i in range(len(X))] * len(results))

        st.subheader("Kết quả dự đoán")
        if choice == "So sánh cả hai mô hình":
            a, b = st.columns(2)
            with a:
                st.metric("LR — số cảnh báo", int((results[0]["Prob_Dropout"] >= results[0]["Threshold_Dropout"]).sum()))
            with b:
                st.metric("CatBoost + FE — số cảnh báo", int((results[1]["Prob_Dropout"] >= results[1]["Threshold_Dropout"]).sum()))
        st.dataframe(predictions, use_container_width=True)
        csv = predictions.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Tải kết quả CSV", data=csv, file_name="ket_qua_du_doan_dropout.csv", mime="text/csv", use_container_width=True)
        st.caption("Xác suất là đầu ra của mô hình đã huấn luyện; không phải xác suất đã hiệu chuẩn trừ khi được đánh giá riêng.")
except Exception as e:
    st.error(f"Không thể xử lý CSV: {e}")
