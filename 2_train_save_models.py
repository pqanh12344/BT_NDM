"""Huan luyen va luu LR Original + CatBoost FE cho kich ban T1 UCI 697.

Mo hinh duoc fit tren toan bo du lieu UCI sau khi giao thuc danh gia da duoc chot.
Nguong demo mac dinh 0.5, dong bo voi run_uci697_fixed.py (khong tuning threshold).
"""
from pathlib import Path
import re
import unicodedata
import json
import numpy as np
import pandas as pd
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CLASSES = ["Dropout", "Enrolled", "Graduate"]
SEED = 42
DEFAULT_THRESHOLD = 0.5
THRESHOLD_FILE = ROOT / "ket_qua_v3" / "thresholds.csv"
ADMIN = {"Debtor", "Tuition fees up to date", "Scholarship holder"}
MACRO = {"Unemployment rate", "Inflation rate", "GDP"}
CAT_ALIASES = {
    "maritalstatus", "applicationmode", "applicationorder", "course",
    "daytimeeveningattendance", "previousqualification", "nacionality",
    "mothersqualification", "fathersqualification", "mothersoccupation",
    "fathersoccupation", "displaced", "educationalspecialneeds", "debtor",
    "tuitionfeesuptodate", "gender", "scholarshipholder", "international"
}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def is_semester2(c):
    n = norm(c)
    return "2ndsem" in n or "secondsem" in n or "semester2" in n



def load_thresholds():
    """Doc nguong tu ket qua run_v3 neu co; neu khong co thi dung 0.5."""
    thresholds = {"lr": DEFAULT_THRESHOLD, "catboost": DEFAULT_THRESHOLD}
    if not THRESHOLD_FILE.exists():
        print(f"Khong tim thay {THRESHOLD_FILE}; dung nguong mac dinh 0.5 cho ca hai mo hinh.")
        return thresholds
    table = pd.read_csv(THRESHOLD_FILE, encoding="utf-8-sig")
    table.columns = [str(c).strip() for c in table.columns]
    norm_cols = {norm(c): c for c in table.columns}
    threshold_col = next((norm_cols[k] for k in ["taustar", "threshold", "thresholdstar", "bestthreshold", "tau", "thresholddropout", "selectedthreshold"] if k in norm_cols), None)
    config_col = next((norm_cols[k] for k in ["config", "modelconfig", "modelname", "model", "cfgname", "experiment", "featuremodel"] if k in norm_cols), None)
    if threshold_col is None or config_col is None:
        print("Khong nhan dien duoc cot model/threshold trong thresholds.csv; dung nguong 0.5.")
        return thresholds
    for _, row in table.iterrows():
        name = str(row[config_col]).strip().lower()
        try:
            value = float(row[threshold_col])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value) or not 0 <= value <= 1:
            continue
        compact = norm(name)
        if "catboost" in compact and ("fe" in compact or "original" not in compact):
            thresholds["catboost"] = value
        elif ("logisticregression" in compact or compact.startswith("lr")) and "fe" not in compact:
            thresholds["lr"] = value
    print(f"Nguong su dung: LR={thresholds['lr']:.4f}; CatBoost+FE={thresholds['catboost']:.4f}")
    return thresholds


def get_data():
    ds = fetch_ucirepo(id=697)
    X = ds.data.features.copy().reset_index(drop=True)
    y = ds.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = y.astype(str).str.strip().reset_index(drop=True)
    if set(y.unique()) != set(CLASSES):
        raise ValueError(f"Nhãn dataset không như mong đợi: {sorted(y.unique())}")
    X.columns = X.columns.astype(str).str.strip()
    # T1: bo bien tai chinh/vi mo chua chac co tai thoi diem canh bao,
    # bo hoc ky 2; giu thong tin tuyen sinh va hoc ky 1.
    sem1 = [c for c in X.columns if "1st sem" in c.lower()]
    cols = [c for c in X.columns if not is_semester2(c) and c not in ADMIN and c not in MACRO]
    if not sem1:
        raise ValueError("Khong tim thay cot hoc ky 1 trong UCI 697.")
    X = X[cols].copy()
    return X, y


def identify_columns(df):
    numeric, categorical = [], []
    for c in df.columns:
        s = df[c]
        if norm(c) in CAT_ALIASES or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_bool_dtype(s):
            categorical.append(c)
        else:
            numeric.append(c)
    return numeric, categorical


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
        raise ValueError("Thieu cot approved/evaluations/enrolled hoc ky 1 de tao FE.")
    approved = pd.to_numeric(Z[approved_c], errors="coerce")
    evaluations = pd.to_numeric(Z[eval_c], errors="coerce")
    enrolled = pd.to_numeric(Z[enrolled_c], errors="coerce")
    Z["FE_approval_rate"] = approved.div(evaluations).where(evaluations.ne(0), 0.0)
    Z["FE_evaluation_rate"] = evaluations.div(enrolled).where(enrolled.ne(0), 0.0)
    Z["FE_completion_rate"] = approved.div(enrolled).where(enrolled.ne(0), 0.0)
    Z["FE_no_eval_1"] = (evaluations == 0).astype(int)
    Z["FE_zero_enrolled_1"] = (enrolled == 0).astype(int)
    return Z.replace([np.inf, -np.inf], np.nan)


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def train_lr(X, y):
    nums, cats = identify_columns(X)
    transformers = []
    if nums:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), nums))
    if cats:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_ohe())
        ]), cats))
    pipe = Pipeline([
        ("preprocess", ColumnTransformer(transformers, remainder="drop")),
        ("model", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs", random_state=SEED))
    ])
    weights = compute_sample_weight(class_weight="balanced", y=y)
    pipe.fit(X, y, model__sample_weight=weights)
    return pipe


def train_catboost(X, y):
    Z = add_fe(X)
    nums, cats = identify_columns(Z)
    # CatBoost nhan categorical duoi dang chuoi; numeric giu numeric va cho phep NaN.
    for c in cats:
        Z[c] = Z[c].where(Z[c].notna(), "__MISSING__").astype(str)
    for c in nums:
        Z[c] = pd.to_numeric(Z[c], errors="coerce")
    model = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        loss_function="MultiClass", random_seed=SEED,
        verbose=False, allow_writing_files=False, thread_count=-1
    )
    weights = compute_sample_weight(class_weight="balanced", y=y)
    model.fit(Z, y, cat_features=cats, sample_weight=weights, verbose=False)
    return {"model": model, "raw_columns": list(X.columns), "feature_columns": list(Z.columns), "categorical_columns": cats}


def main():
    thresholds = load_thresholds()
    X, y = get_data()
    print(f"Dataset T1: {X.shape[0]} mau, {X.shape[1]} cot goc")
    print("Phan bo nhan:", y.value_counts().to_dict())
    print("Dang train Logistic Regression (Original)...")
    lr = train_lr(X, y)
    joblib.dump({"model": lr, "raw_columns": list(X.columns), "classes": CLASSES,
                 "threshold": thresholds["lr"], "feature_set": "Original"},
                MODEL_DIR / "logistic_regression.joblib")
    print("Dang train CatBoost + FE...")
    cb_bundle = train_catboost(X, y)
    cb_bundle.update({"classes": CLASSES, "threshold": thresholds["catboost"], "feature_set": "Original + FE"})
    joblib.dump(cb_bundle, MODEL_DIR / "catboost_fe.joblib")
    metadata = {
        "dataset": "UCI 697 - Predict Students' Dropout and Academic Success",
        "scenario": "T1: admission/demographic + first-semester features; no second-semester features",
        "classes": CLASSES,
        "dropout_thresholds": thresholds,
        "threshold_policy": "Read from ket_qua_v3/thresholds.csv when recognized; otherwise fixed fallback 0.5",
        "models": ["Logistic Regression (Original)", "CatBoost + FE"]
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Da luu models/logistic_regression.joblib")
    print("Da luu models/catboost_fe.joblib")
    print("Da luu models/metadata.json")
    print("Luu y: chi chay file nay khi can train lai; Streamlit khong train model.")


if __name__ == "__main__":
    main()
