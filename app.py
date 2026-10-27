
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Canh bao som nguy co bo hoc",
    page_icon="🎓",
    layout="wide",
)

DATA_PATH = Path("data/data.csv")
RESULT_DIR = Path("ket_qua_v3")
THRESHOLD_PATH = RESULT_DIR / "thresholds.csv"
ARTIFACT_DIR = RESULT_DIR / "artifacts"

LR_PATH = ARTIFACT_DIR / "logistic_regression.joblib"
CAT_PATH = ARTIFACT_DIR / "catboost_fe.joblib"

RANDOM_STATE = 42
CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]

FE_COLUMNS = [
    "FE_approval_rate",
    "FE_evaluation_rate",
    "FE_completion_rate",
    "FE_no_eval_1",
    "FE_zero_enrolled_1",
]

CATEGORICAL_NAMES = {
    "maritalstatus", "applicationmode", "applicationorder",
    "course", "daytimeeveningattendance", "previousqualification",
    "nationality", "nacionality", "mothersqualification",
    "fathersqualification", "mothersoccupation", "fathersoccupation",
    "displaced", "educationalspecialneeds", "debtor",
    "tuitionfeesuptodate", "gender", "scholarshipholder",
    "international",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def norm_name(value):
    return "".join(
        ch.lower() for ch in str(value) if ch.isalnum()
    )


def find_column(columns, candidates):
    lookup = {norm_name(c): c for c in columns}
    for candidate in candidates:
        if norm_name(candidate) in lookup:
            return lookup[norm_name(candidate)]
    return None


def remove_second_semester(X):
    def is_hk2(col):
        name = norm_name(col)
        return (
            "2ndsem" in name
            or "2ndsemester" in name
            or "semester2" in name
            or "hk2" in name
        )

    return X.drop(
        columns=[c for c in X.columns if is_hk2(c)],
        errors="ignore",
    )


def identify_categorical_columns(columns):
    return [
        c for c in columns
        if norm_name(c) in CATEGORICAL_NAMES
    ]


def read_csv_flexible(source):
    return pd.read_csv(source, sep=None, engine="python")


def load_training_data():
    if DATA_PATH.exists():
        df = read_csv_flexible(DATA_PATH)
        df.columns = [str(c).strip() for c in df.columns]

        target_col = find_column(df.columns, ["Target"])
        if target_col is None:
            raise ValueError(
                f"{DATA_PATH} không có cột Target."
            )

        y = df[target_col].astype(str).str.strip()
        X = df.drop(columns=[target_col])

    else:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=697)
        X = dataset.data.features.copy()
        y = dataset.data.targets.iloc[:, 0].astype(str).str.strip()
        X.columns = [str(c).strip() for c in X.columns]

    X.columns = [str(c).strip() for c in X.columns]
    X = remove_second_semester(X)

    valid = y.isin(CLASS_NAMES)
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    if set(y.unique()) != set(CLASS_NAMES):
        raise ValueError(
            f"Dữ liệu phải có đủ các lớp {CLASS_NAMES}. "
            f"Các lớp hiện có: {sorted(y.unique().tolist())}"
        )

    return X, y


def coerce_features(X, feature_columns):
    X = X.copy()

    missing = [
        c for c in feature_columns
        if c not in X.columns
    ]

    if missing:
        raise ValueError(
            "CSV thiếu các cột bắt buộc: "
            + ", ".join(missing[:15])
        )

    X = X.loc[:, feature_columns].copy()

    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X


def add_feature_engineering(X):
    X = X.copy()

    approved_col = find_column(
        X.columns,
        ["Curricular units 1st sem (approved)"],
    )
    evaluated_col = find_column(
        X.columns,
        ["Curricular units 1st sem (evaluations)"],
    )
    enrolled_col = find_column(
        X.columns,
        ["Curricular units 1st sem (enrolled)"],
    )

    if not all([approved_col, evaluated_col, enrolled_col]):
        raise ValueError(
            "Không tìm thấy đủ ba cột học kỳ 1 để tạo FE: "
            "approved, evaluations, enrolled."
        )

    approved = pd.to_numeric(
        X[approved_col], errors="coerce"
    )
    evaluated = pd.to_numeric(
        X[evaluated_col], errors="coerce"
    )
    enrolled = pd.to_numeric(
        X[enrolled_col], errors="coerce"
    )

    X["FE_approval_rate"] = (
        approved / evaluated.replace(0, np.nan)
    )
    X["FE_evaluation_rate"] = (
        evaluated / enrolled.replace(0, np.nan)
    )
    X["FE_completion_rate"] = (
        approved / enrolled.replace(0, np.nan)
    )
    X["FE_no_eval_1"] = evaluated.eq(0).astype(int)
    X["FE_zero_enrolled_1"] = enrolled.eq(0).astype(int)

    return X


def prepare_catboost_data(X, feature_columns, categorical_columns):
    X = X.loc[:, feature_columns].copy()

    for col in categorical_columns:
        X[col] = (
            X[col]
            .where(X[col].notna(), "__MISSING__")
            .astype(str)
        )

    return X


# ============================================================
# THRESHOLDS
# ============================================================

def load_thresholds():
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {THRESHOLD_PATH}. "
            "Hãy chạy run_v3.py để tạo ngưỡng thực nghiệm trước."
        )

    df = read_csv_flexible(THRESHOLD_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    config_col = find_column(
        df.columns,
        ["Config", "Model", "Model_Name"],
    )
    threshold_col = find_column(
        df.columns,
        [
            "tau_star",
            "threshold",
            "threshold_star",
            "best_threshold",
            "tau",
        ],
    )

    if config_col is None or threshold_col is None:
        raise ValueError(
            "Không nhận diện được cột tên mô hình hoặc cột ngưỡng "
            f"trong thresholds.csv. Các cột hiện có: {df.columns.tolist()}"
        )

    df["_normalized_config"] = (
        df[config_col].astype(str).map(norm_name)
    )

    aliases = {
        "Logistic Regression": {
            norm_name("Logistic Regression"),
            norm_name("Logistic Regression Original"),
            norm_name("LR"),
        },
        "CatBoost + FE": {
            norm_name("CatBoost + FE"),
            norm_name("CatBoost FE"),
        },
    }

    result = {}

    for model_name, names in aliases.items():
        rows = df[df["_normalized_config"].isin(names)]

        if rows.empty:
            raise ValueError(
                f"Không tìm thấy ngưỡng cho {model_name}. "
                f"Tên cấu hình có trong file: {df[config_col].tolist()}"
            )

        tau = pd.to_numeric(
            rows.iloc[0][threshold_col],
            errors="coerce",
        )

        if pd.isna(tau) or not 0 < float(tau) < 1:
            raise ValueError(
                f"Ngưỡng của {model_name} không hợp lệ: {tau}"
            )

        result[model_name] = float(tau)

    return result


# ============================================================
# MODEL 1: LOGISTIC REGRESSION - ORIGINAL
# MODEL 2: CATBOOST + FE
# ============================================================

def build_lr_pipeline(feature_columns, categorical_columns):
    numeric_cols = [
        c for c in feature_columns
        if c not in categorical_columns
    ]

    categorical_cols = [
        c for c in feature_columns
        if c in categorical_columns
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=3000,
            random_state=RANDOM_STATE,
        )),
    ])


def train_and_save_models():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_training_data()
    raw_features = list(X.columns)
    categorical_columns = identify_categorical_columns(
        raw_features
    )

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    # Model 1: Logistic Regression - Original
    X_lr = coerce_features(X_train, raw_features)

    lr_model = build_lr_pipeline(
        raw_features,
        categorical_columns,
    )

    lr_model.fit(
        X_lr,
        y_train,
        classifier__sample_weight=weights,
    )

    joblib.dump(
        {
            "model": lr_model,
            "raw_features": raw_features,
            "features": raw_features,
            "categorical_columns": categorical_columns,
            "use_fe": False,
        },
        LR_PATH,
    )

    # Model 2: CatBoost + FE
    X_cat = add_feature_engineering(
        coerce_features(X_train, raw_features)
    )

    cat_features = raw_features + FE_COLUMNS

    X_cat = prepare_catboost_data(
        X_cat,
        cat_features,
        categorical_columns,
    )

    cat_model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )

    cat_model.fit(
        X_cat,
        y_train,
        cat_features=categorical_columns,
        sample_weight=weights,
        verbose=False,
    )

    joblib.dump(
        {
            "model": cat_model,
            "raw_features": raw_features,
            "features": cat_features,
            "categorical_columns": categorical_columns,
            "use_fe": True,
        },
        CAT_PATH,
    )


@st.cache_resource(show_spinner="Đang nạp mô hình...")
def load_models():
    thresholds = load_thresholds()

    if not LR_PATH.exists() or not CAT_PATH.exists():
        train_and_save_models()

    return (
        joblib.load(LR_PATH),
        joblib.load(CAT_PATH),
        thresholds,
    )


# ============================================================
# PREDICTION
# ============================================================

def get_probabilities(model, X):
    probs = model.predict_proba(X)

    if isinstance(model, Pipeline):
        classes = list(
            model.named_steps["classifier"].classes_
        )
    else:
        classes = list(model.classes_)

    aligned = np.zeros(
        (len(X), len(CLASS_NAMES)),
        dtype=float,
    )

    for j, label in enumerate(classes):
        if label in CLASS_NAMES:
            aligned[:, CLASS_NAMES.index(label)] = probs[:, j]

    return aligned


def predict_one_model(bundle, input_df, threshold, model_name):
    raw_features = bundle["raw_features"]
    feature_columns = bundle["features"]
    categorical_columns = bundle["categorical_columns"]

    X = coerce_features(input_df, raw_features)

    if bundle["use_fe"]:
        X = add_feature_engineering(X)

    X = X.loc[:, feature_columns]
    model = bundle["model"]

    if isinstance(model, CatBoostClassifier):
        X = prepare_catboost_data(
            X,
            feature_columns,
            categorical_columns,
        )

    probabilities = get_probabilities(model, X)

    p_dropout = probabilities[:, CLASS_NAMES.index("Dropout")]
    predicted_class = [
        CLASS_NAMES[i]
        for i in probabilities.argmax(axis=1)
    ]

    prefix = (
        "LR"
        if model_name == "Logistic Regression"
        else "CatBoost"
    )

    return pd.DataFrame({
        f"P_Dropout_{prefix}": p_dropout,
        f"P_Enrolled_{prefix}": probabilities[
            :, CLASS_NAMES.index("Enrolled")
        ],
        f"P_Graduate_{prefix}": probabilities[
            :, CLASS_NAMES.index("Graduate")
        ],
        f"Predicted_Class_{prefix}": predicted_class,
        f"Threshold_{prefix}": threshold,
        f"Risk_{prefix}": np.where(
            p_dropout >= threshold,
            "Được mô hình đánh dấu có nguy cơ bỏ học",
            "Chưa vượt ngưỡng cảnh báo",
        ),
    })


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🎓 Hệ thống cảnh báo sớm nguy cơ bỏ học")
st.caption("Bộ dữ liệu UCI 697 · Kịch bản T1 — sau học kỳ 1")

st.info(
    "Hệ thống hỗ trợ sàng lọc để xem xét nhu cầu hỗ trợ của sinh viên. "
    "Cảnh báo không có nghĩa sinh viên chắc chắn sẽ bỏ học."
)

with st.expander("Hướng dẫn sử dụng", expanded=True):
    st.markdown(
        """
        1. Tải lên file CSV chứa danh sách sinh viên.
        2. Chọn một mô hình hoặc so sánh cả hai.
        3. Xem xác suất ba lớp và cảnh báo.
        4. Tải kết quả dự đoán xuống CSV.

        File đầu vào không bắt buộc có cột `Target`. Các cột phải
        khớp với schema dữ liệu huấn luyện. Đặc trưng học kỳ 2
        sẽ không được sử dụng.
        """
    )

try:
    lr_bundle, cat_bundle, thresholds = load_models()
except Exception as exc:
    st.error(f"Không thể khởi tạo mô hình: {exc}")
    st.warning(
        "Kiểm tra thresholds.csv, dữ liệu huấn luyện và các thư viện. "
        "Nếu đã đổi pipeline, hãy đồng bộ cách tiền xử lý trước khi chạy demo."
    )
    st.stop()

st.subheader("Bước 1 — Tải file CSV")

uploaded_file = st.file_uploader(
    "Chọn file CSV chứa nhiều sinh viên",
    type=["csv"],
)

st.subheader("Bước 2 — Chọn mô hình")

model_choice = st.radio(
    "Mô hình sử dụng",
    [
        "So sánh cả hai mô hình",
        "Logistic Regression",
        "CatBoost + FE",
    ],
    horizontal=True,
)

if uploaded_file is None:
    st.caption("Vui lòng tải lên file CSV để bắt đầu.")
else:
    try:
        input_df = read_csv_flexible(uploaded_file)
        input_df.columns = [
            str(c).strip() for c in input_df.columns
        ]

        id_col = find_column(
            input_df.columns,
            ["Student_ID", "Student ID", "ID", "MSSV"],
        )

        if id_col is not None:
            student_ids = (
                input_df[id_col].astype(str).reset_index(drop=True)
            )
        else:
            student_ids = pd.Series(
                [
                    f"SV_{i + 1:04d}"
                    for i in range(len(input_df))
                ],
                name="Student_ID",
            )

        # Khong dung Target de du doan.
        target_col = find_column(input_df.columns, ["Target"])
        if target_col is not None:
            input_df = input_df.drop(columns=[target_col])

        # Loai cot HK2 neu co.
        input_df = remove_second_semester(input_df)

        st.write(f"Đã đọc **{len(input_df):,}** bản ghi.")
        st.dataframe(
            input_df.head(5),
            use_container_width=True,
        )

        if st.button(
            "Chạy dự đoán",
            type="primary",
            use_container_width=True,
        ):
            run_lr = model_choice in [
                "So sánh cả hai mô hình",
                "Logistic Regression",
            ]
            run_cat = model_choice in [
                "So sánh cả hai mô hình",
                "CatBoost + FE",
            ]

            result = pd.DataFrame({
                "Student_ID": student_ids
            })

            with st.spinner("Đang chạy dự đoán..."):
                if run_lr:
                    lr_result = predict_one_model(
                        lr_bundle,
                        input_df,
                        thresholds["Logistic Regression"],
                        "Logistic Regression",
                    )
                    result = pd.concat(
                        [
                            result,
                            lr_result.reset_index(drop=True),
                        ],
                        axis=1,
                    )

                if run_cat:
                    cat_result = predict_one_model(
                        cat_bundle,
                        input_df,
                        thresholds["CatBoost + FE"],
                        "CatBoost + FE",
                    )
                    result = pd.concat(
                        [
                            result,
                            cat_result.reset_index(drop=True),
                        ],
                        axis=1,
                    )

            st.success("Đã hoàn thành dự đoán.")

            st.subheader("Bước 3 — Tổng quan cảnh báo")

            n_metrics = 1 + int(run_lr) + int(run_cat)
            metric_cols = st.columns(n_metrics)

            metric_cols[0].metric(
                "Số sinh viên",
                f"{len(result):,}",
            )

            metric_idx = 1

            if run_lr:
                count_lr = (
                    result["Risk_LR"]
                    == "Được mô hình đánh dấu có nguy cơ bỏ học"
                ).sum()

                metric_cols[metric_idx].metric(
                    "Cảnh báo — Logistic Regression",
                    f"{count_lr:,}",
                )
                metric_idx += 1

            if run_cat:
                count_cat = (
                    result["Risk_CatBoost"]
                    == "Được mô hình đánh dấu có nguy cơ bỏ học"
                ).sum()

                metric_cols[metric_idx].metric(
                    "Cảnh báo — CatBoost + FE",
                    f"{count_cat:,}",
                )

            st.subheader("Bước 4 — Kết quả từng sinh viên")
            st.dataframe(
                result,
                use_container_width=True,
                height=420,
            )

            csv_result = result.to_csv(
                index=False,
                encoding="utf-8-sig",
            )

            st.download_button(
                "⬇️ Tải kết quả dự đoán CSV",
                data=csv_result,
                file_name="ket_qua_du_doan_uci697.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.caption(
                "Ngưỡng được lấy từ thresholds.csv. "
                "P_Dropout là xác suất mô hình ước lượng cho lớp Dropout; "
                "không phải khẳng định chắc chắn về tương lai sinh viên."
            )

    except Exception as exc:
        st.error(f"Không thể đọc hoặc dự đoán file CSV: {exc}")
        st.markdown(
            """
            **Hãy kiểm tra:**
            - File CSV có đủ các cột đầu vào bắt buộc.
            - Tên cột khớp với dữ liệu UCI 697.
            - Có ba cột học kỳ 1 dùng tạo FE: `approved`,
              `evaluations`, `enrolled`.
            - Pipeline demo đã được đồng bộ với pipeline huấn luyện.
            """
        )
