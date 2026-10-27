# ============================================================
# BÀI TẬP CUỐI KỲ - MÔN NHẬN DẠNG MẪU
# ĐỀ TÀI: DỰ ĐOÁN VÀ CẢNH BÁO SỚM NGUY CƠ BỎ HỌC CỦA SINH VIÊN
# DATASET: UCI 697 - Predict Students' Dropout and Academic Success
#
# Bản v3: giữ nguyên dữ liệu, FE, tiền xử lý và 5 mô hình của run_v2.py; đổi giao thức đánh giá theo
# slide 6 (Hồi quy logistic) và mục IV.7 của báo cáo:
#   1. Chia Train/Test 80:20 phân tầng; Test khóa đến bước 4.
#   2. Stratified 5-fold CV trên Train, lưu dự đoán out-of-fold (OOF).
#   3. Mỗi cấu hình chọn ngưỡng τ* tối đa F2 (lớp Dropout) trên OOF của Train, rồi khóa.
#   4. Học lại trên toàn bộ Train, đánh giá Test một lần tại τ*.
# Thêm: Specificity, log loss, Brier, kiểm định Nadeau–Bengio trên CV, bootstrap ghép cặp trên Test,
# odds ratio của Logistic Regression nhị phân, phân tích lỗi và Hình 7–15 của báo cáo.
#
# Chạy:
#   python run_v3.py                         # tải dữ liệu qua ucimlrepo
#   python run_v3.py --data path/data.csv    # hoặc đọc file data.csv tải từ UCI (dấu ;)
# Kết quả ghi vào thư mục ket_qua_v3/ cạnh script.
# ============================================================

import argparse
import json
import re
import unicodedata
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# ============================================================
# 1. CẤU HÌNH
# ============================================================

SEED = 42
N_SPLITS = 5
TEST_SIZE = 0.2
N_BOOT = 1000
TAUS = np.round(np.arange(0.05, 0.951, 0.01), 2)
DROPOUT_LABEL = "Dropout"
BASE = ("Original", "Logistic Regression")
MAIN = ("Original + FE", "CatBoost")
ALT = ("Original + FE", "LightGBM")
FEATURE_ORDER = ["Original", "Original + FE"]
MODEL_ORDER = ["Dummy Prior", "Logistic Regression", "Random Forest", "CatBoost", "LightGBM"]

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="Đường dẫn data.csv của UCI 697; bỏ trống thì tải qua ucimlrepo")
args = parser.parse_args()
OUT = Path(__file__).resolve().parent / "ket_qua_v3"
(OUT / "hinh").mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

# ============================================================
# 2. DỮ LIỆU (giống run_v2.py)
# ============================================================

if args.data:
    raw = pd.read_csv(args.data, sep=";", encoding="utf-8-sig")
    raw.columns = [c.strip() for c in raw.columns]
    X_raw, y_raw = raw.drop(columns=["Target"]), raw["Target"]
else:
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=697)
    X_raw = dataset.data.features.copy()
    y_raw = dataset.data.targets.iloc[:, 0]
X_raw = X_raw.reset_index(drop=True)
y_raw = y_raw.reset_index(drop=True).astype(str).str.strip()

class_names = sorted(y_raw.unique())
class_to_id = {name: idx for idx, name in enumerate(class_names)}
y = y_raw.map(class_to_id).to_numpy(dtype=int)
dropout_id = class_to_id[DROPOUT_LABEL]
n_classes = len(class_names)
print(f"Dữ liệu: {X_raw.shape}, lớp: {class_names}")


def normalize_name(name):
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", text.lower())


second_sem_cols = [c for c in X_raw.columns
                   if any(k in normalize_name(c) for k in ("2ndsem", "secondsem", "semester2"))]
X = X_raw.drop(columns=second_sem_cols).copy()

categorical_aliases = {
    "maritalstatus", "applicationmode", "applicationorder", "course", "daytimeeveningattendance",
    "previousqualification", "nacionality", "mothersqualification", "fathersqualification",
    "mothersoccupation", "fathersoccupation", "displaced", "educationalspecialneeds", "debtor",
    "tuitionfeesuptodate", "gender", "scholarshipholder", "international",
}


def identify_columns(df):
    cats = [c for c in df.columns if normalize_name(c) in categorical_aliases
            or pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
    return [c for c in df.columns if c not in cats], cats


def col(df, name):
    return {normalize_name(c): c for c in df.columns}[normalize_name(name)]


approved = pd.to_numeric(X[col(X, "Curricular units 1st sem (approved)")])
evaluations = pd.to_numeric(X[col(X, "Curricular units 1st sem (evaluations)")])
enrolled = pd.to_numeric(X[col(X, "Curricular units 1st sem (enrolled)")])

X_fe = X.copy()
X_fe["FE_approval_rate"] = approved.div(evaluations).where(evaluations.ne(0), 0.0)
X_fe["FE_evaluation_rate"] = evaluations.div(enrolled).where(enrolled.ne(0), 0.0)
X_fe["FE_completion_rate"] = approved.div(enrolled).where(enrolled.ne(0), 0.0)
X_fe["FE_no_eval_1"] = (evaluations == 0).astype(int)
X_fe["FE_zero_enrolled_1"] = (enrolled == 0).astype(int)
feature_sets = {"Original": X.copy(), "Original + FE": X_fe}
print(f"Bỏ {len(second_sem_cols)} cột học kỳ 2; Original {X.shape[1]} cột, Original + FE {X_fe.shape[1]} cột")

# ============================================================
# 3. MÔ HÌNH VÀ TIỀN XỬ LÝ (giống run_v2.py)
# ============================================================


def make_preprocessor(df, scale):
    nums, cats = identify_columns(df)
    num_steps = [("imputer", SimpleImputer(strategy="median"))] + ([("scaler", StandardScaler())] if scale else [])
    return ColumnTransformer([
        ("numeric", Pipeline(num_steps), nums),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats),
    ])


def make_model(name):
    return {
        "Dummy Prior": lambda: DummyClassifier(strategy="prior"),
        "Logistic Regression": lambda: LogisticRegression(C=1.0, max_iter=3000, random_state=SEED),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=1,
                                                        random_state=SEED, n_jobs=-1),
        "CatBoost": lambda: CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, loss_function="MultiClass",
                                               random_seed=SEED, verbose=False, allow_writing_files=False,
                                               thread_count=-1),
        "LightGBM": lambda: LGBMClassifier(objective="multiclass", num_class=n_classes, n_estimators=500,
                                           learning_rate=0.05, num_leaves=31, random_state=SEED, verbosity=-1,
                                           n_jobs=-1),
    }[name]()


def fit_predict(model_name, X_tr, y_tr, X_te):
    """Học trên (X_tr, y_tr), trả về xác suất ba lớp của X_te theo thứ tự class_names."""
    model = make_model(model_name)
    weights = compute_sample_weight("balanced", y_tr)
    if model_name == "Dummy Prior":
        model.fit(X_tr, y_tr)
        proba, classes = model.predict_proba(X_te), model.classes_
    elif model_name == "CatBoost":
        nums, cats = identify_columns(X_tr)
        tr, te = X_tr.copy(), X_te.copy()
        for c in nums:
            med = tr[c].median()
            tr[c], te[c] = tr[c].fillna(med), te[c].fillna(med)
        for c in cats:
            tr[c], te[c] = tr[c].astype(str), te[c].astype(str)
        model.fit(tr, y_tr, cat_features=cats, sample_weight=weights, verbose=False)
        proba, classes = model.predict_proba(te), model.classes_
    else:
        pipe = Pipeline([("preprocess", make_preprocessor(X_tr, scale=model_name == "Logistic Regression")),
                         ("model", model)])
        pipe.fit(X_tr, y_tr, model__sample_weight=weights)
        proba, classes = pipe.predict_proba(X_te), pipe.named_steps["model"].classes_
    aligned = np.zeros((len(X_te), n_classes))
    for j, c in enumerate(classes):
        aligned[:, int(c)] = proba[:, j]
    return aligned


# ============================================================
# 4. CHỈ SỐ
# ============================================================


def f2_at(yb, p, tau):
    return fbeta_score(yb, (p >= tau).astype(int), beta=2, zero_division=0)


def best_tau(yb, p):
    scores = [f2_at(yb, p, t) for t in TAUS]
    return float(TAUS[int(np.argmax(scores))])


def metrics(y3, proba, tau):
    yb = (y3 == dropout_id).astype(int)
    p = proba[:, dropout_id]
    pred = (p >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(yb, pred, labels=[0, 1]).ravel()
    arg = proba.argmax(1)
    return {
        "F2": fbeta_score(yb, pred, beta=2, zero_division=0), "Recall": recall_score(yb, pred, zero_division=0),
        "Precision": precision_score(yb, pred, zero_division=0), "Specificity": tn / (tn + fp),
        "F1": f1_score(yb, pred, zero_division=0), "AP": average_precision_score(yb, p),
        "ROC_AUC": roc_auc_score(yb, p), "Brier": brier_score_loss(yb, p),
        "LogLoss": log_loss(y3, np.clip(proba, 1e-15, 1), labels=list(range(n_classes))),
        "Accuracy": accuracy_score(y3, arg), "Balanced_Accuracy": balanced_accuracy_score(y3, arg),
        "Macro_F1": f1_score(y3, arg, average="macro"), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def cfg_name(cfg):
    fs, m = cfg
    return m + (" + FE" if fs == "Original + FE" else "")


# ============================================================
# 5. CHIA TRAIN/TEST VÀ CROSS-VALIDATION TRÊN TRAIN
# ============================================================

idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=SEED)
y_train, y_test = y[idx_train], y[idx_test]
yb_train, yb_test = (y_train == dropout_id).astype(int), (y_test == dropout_id).astype(int)
folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(idx_train, y_train))
ratio = len(folds[0][1]) / len(folds[0][0])
print(f"Train {len(idx_train)} (Dropout {yb_train.sum()}), Test {len(idx_test)} (Dropout {yb_test.sum()})")

configs = [(fs, m) for fs in FEATURE_ORDER for m in MODEL_ORDER]
oof, test_proba, fold_rows = {}, {}, []
for cfg in configs:
    fs, m = cfg
    Xs = feature_sets[fs]
    X_tr_all, X_te = Xs.iloc[idx_train].reset_index(drop=True), Xs.iloc[idx_test].reset_index(drop=True)
    P = np.zeros((len(idx_train), n_classes))
    for k, (tr, va) in enumerate(folds, start=1):
        P[va] = fit_predict(m, X_tr_all.iloc[tr], y_train[tr], X_tr_all.iloc[va])
    oof[cfg] = P
    for k, (tr, va) in enumerate(folds, start=1):
        # Ngưỡng cross-fit: chọn trên OOF của 4 fold còn lại, áp vào fold k
        tau_k = best_tau(yb_train[tr], P[tr, dropout_id])
        r = metrics(y_train[va], P[va], 0.5)
        fold_rows.append({"Feature_Set": fs, "Model": m, "Fold": k, "tau_crossfit": tau_k,
                          "F2_crossfit": f2_at(yb_train[va], P[va, dropout_id], tau_k),
                          "F2_at_0.5": r["F2"], "AP": r["AP"], "ROC_AUC": r["ROC_AUC"], "Brier": r["Brier"],
                          "LogLoss": r["LogLoss"], "Macro_F1": r["Macro_F1"]})
    test_proba[cfg] = fit_predict(m, X_tr_all, y_train, X_te)
    print(f"  xong {cfg_name(cfg)}")

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(OUT / "cv_fold_results.csv", index=False, encoding="utf-8-sig")

# ============================================================
# 6. NGƯỠNG τ* VÀ KẾT QUẢ TRÊN TEST
# ============================================================

tau_star = {cfg: best_tau(yb_train, oof[cfg][:, dropout_id]) for cfg in configs}
thr_rows, test_rows = [], []
for cfg in configs:
    p_oof = oof[cfg][:, dropout_id]
    thr_rows.append({"Config": cfg_name(cfg), "tau_star": tau_star[cfg],
                     "F2_OOF_at_tau": f2_at(yb_train, p_oof, tau_star[cfg]), "F2_OOF_at_0.5": f2_at(yb_train, p_oof, 0.5),
                     "Recall_OOF_at_tau": recall_score(yb_train, (p_oof >= tau_star[cfg]).astype(int)),
                     "Precision_OOF_at_tau": precision_score(yb_train, (p_oof >= tau_star[cfg]).astype(int),
                                                             zero_division=0)})
    test_rows.append({"Config": cfg_name(cfg), "tau_star": tau_star[cfg], **metrics(y_test, test_proba[cfg], tau_star[cfg])})
thr_df, test_df = pd.DataFrame(thr_rows), pd.DataFrame(test_rows)
thr_df.to_csv(OUT / "thresholds.csv", index=False, encoding="utf-8-sig")
test_df.to_csv(OUT / "test_metrics.csv", index=False, encoding="utf-8-sig")

cv_summary = fold_df.groupby(["Feature_Set", "Model"], sort=False)[
    ["AP", "ROC_AUC", "LogLoss", "Brier", "Macro_F1", "F2_crossfit", "F2_at_0.5"]].agg(["mean", "std"])
cv_summary.columns = [f"{a}_{b}" for a, b in cv_summary.columns]
cv_summary.reset_index().to_csv(OUT / "cv_summary.csv", index=False, encoding="utf-8-sig")

# ============================================================
# 7. SO SÁNH VỚI BASE: Nadeau–Bengio trên CV, bootstrap ghép cặp trên Test
# ============================================================


def nadeau_bengio(a, b):
    d = np.asarray(a) - np.asarray(b)
    v = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / len(d) + ratio) * v) if v > 0 else 0.0
    return float(d.mean()), float(2 * stats.t.sf(abs(t), len(d) - 1)), int((d > 0).sum())


def fold_metric(cfg, k):
    return fold_df[(fold_df.Feature_Set == cfg[0]) & (fold_df.Model == cfg[1])].sort_values("Fold")[k].to_numpy()


rng = np.random.default_rng(SEED)
boot_idx = rng.integers(0, len(y_test), size=(N_BOOT, len(y_test)))


def boot_delta(cfg_a, cfg_b):
    pa, pb = test_proba[cfg_a][:, dropout_id], test_proba[cfg_b][:, dropout_id]
    ta, tb = tau_star[cfg_a], tau_star[cfg_b]
    dF2, dAP = [], []
    for ix in boot_idx:
        yy = yb_test[ix]
        dF2.append(f2_at(yy, pa[ix], ta) - f2_at(yy, pb[ix], tb))
        dAP.append(average_precision_score(yy, pa[ix]) - average_precision_score(yy, pb[ix]))
    return np.percentile(dF2, [2.5, 97.5]), np.percentile(dAP, [2.5, 97.5])


def compare(cfg_a, cfg_b):
    ta = test_df.set_index("Config").loc[cfg_name(cfg_a)]
    tb = test_df.set_index("Config").loc[cfg_name(cfg_b)]
    dF2_cv = nadeau_bengio(fold_metric(cfg_a, "F2_crossfit"), fold_metric(cfg_b, "F2_crossfit"))
    dAP_cv = nadeau_bengio(fold_metric(cfg_a, "AP"), fold_metric(cfg_b, "AP"))
    ci_f2, ci_ap = boot_delta(cfg_a, cfg_b)
    return {"dF2_CV": dF2_cv[0], "p_F2_CV": dF2_cv[1], "wins_F2_CV": dF2_cv[2],
            "dAP_CV": dAP_cv[0], "p_AP_CV": dAP_cv[1], "wins_AP_CV": dAP_cv[2],
            "dF2_Test": ta.F2 - tb.F2, "dF2_Test_CI_low": ci_f2[0], "dF2_Test_CI_high": ci_f2[1],
            "dAP_Test": ta.AP - tb.AP, "dAP_Test_CI_low": ci_ap[0], "dAP_Test_CI_high": ci_ap[1]}


cmp_rows = [{"Config": cfg_name(c), **compare(c, BASE)}
            for c in [MAIN, ALT, ("Original + FE", "Random Forest"), ("Original + FE", "Logistic Regression"),
                      ("Original", "CatBoost")]]
pd.DataFrame(cmp_rows).to_csv(OUT / "comparison_vs_base.csv", index=False, encoding="utf-8-sig")
abl_rows = [{"Model": m, **compare(("Original + FE", m), ("Original", m))} for m in MODEL_ORDER[1:]]
pd.DataFrame(abl_rows).to_csv(OUT / "ablation_fe.csv", index=False, encoding="utf-8-sig")

# ============================================================
# 8. ODDS RATIO: Logistic Regression nhị phân Dropout so với phần còn lại, học trên Train
# ============================================================

X_tr_orig = X.iloc[idx_train].reset_index(drop=True)
or_pipe = Pipeline([("preprocess", make_preprocessor(X_tr_orig, scale=True)),
                    ("model", LogisticRegression(C=1.0, max_iter=3000, random_state=SEED))])
or_pipe.fit(X_tr_orig, yb_train, model__sample_weight=compute_sample_weight("balanced", yb_train))
names = or_pipe.named_steps["preprocess"].get_feature_names_out()
coef = or_pipe.named_steps["model"].coef_[0]
or_df = pd.DataFrame({"Feature": [n.split("__", 1)[1] for n in names], "w": coef, "OR": np.exp(coef)})
or_df = or_df.reindex(or_df.w.abs().sort_values(ascending=False).index).reset_index(drop=True)
or_df.to_csv(OUT / "odds_ratio.csv", index=False, encoding="utf-8-sig")

# ============================================================
# 9. PHÂN TÍCH LỖI
# ============================================================

Xtr_raw = X.iloc[idx_train].reset_index(drop=True)
ap1 = pd.to_numeric(Xtr_raw[col(X, "Curricular units 1st sem (approved)")])
ev1 = pd.to_numeric(Xtr_raw[col(X, "Curricular units 1st sem (evaluations)")])
groups = {
    "evaluations HK1 = 0": ev1 == 0,
    "approved HK1 = 0": ap1 == 0,
    "approved HK1 ≥ 5": ap1 >= 5,
    "Tuition fees up to date = 0": Xtr_raw[col(X, "Tuition fees up to date")] == 0,
    "Debtor = 1": Xtr_raw[col(X, "Debtor")] == 1,
    "Tuổi nhập học ≥ 26": Xtr_raw[col(X, "Age at enrollment")] >= 26,
}
miss = {cfg: (oof[cfg][:, dropout_id] < tau_star[cfg]) & (yb_train == 1) for cfg in (BASE, MAIN)}
grp_rows = []
for g, mask in groups.items():
    mk = mask.to_numpy() & (yb_train == 1)
    grp_rows.append({"Group": g, "n_Dropout": int(mk.sum()),
                     "miss_rate_Base": (miss[BASE] & mk).sum() / max(mk.sum(), 1),
                     "miss_rate_CatBoost_FE": (miss[MAIN] & mk).sum() / max(mk.sum(), 1)})
grp_df = pd.DataFrame(grp_rows)
grp_df.to_csv(OUT / "error_groups_train_oof.csv", index=False, encoding="utf-8-sig")

err = {}
for cfg in (BASE, MAIN):
    p = test_proba[cfg][:, dropout_id]
    pred = (p >= tau_star[cfg]).astype(int)
    fn_mask = (pred == 0) & (yb_test == 1)
    fp_mask = (pred == 1) & (yb_test == 0)
    err[cfg_name(cfg)] = {
        "cm3_test": confusion_matrix(y_test, test_proba[cfg].argmax(1), labels=list(range(n_classes))).tolist(),
        "FP_true_class": pd.Series(np.array(class_names)[y_test[fp_mask]]).value_counts().to_dict(),
        "FN_near_tau": int((fn_mask & (p >= tau_star[cfg] - 0.1)).sum()),
        "FN_far": int((fn_mask & (p < tau_star[cfg] / 2)).sum()), "FN": int(fn_mask.sum()),
    }
wrong_b = (test_proba[BASE][:, dropout_id] < tau_star[BASE]) & (yb_test == 1)
wrong_m = (test_proba[MAIN][:, dropout_id] < tau_star[MAIN]) & (yb_test == 1)
b01, b10 = int((wrong_b & ~wrong_m).sum()), int((~wrong_b & wrong_m).sum())
err["mcnemar_dropout_test"] = {"base_wrong_main_right": b01, "base_right_main_wrong": b10,
                               "p": float(stats.binomtest(b01, b01 + b10).pvalue) if b01 + b10 else 1.0}
good = (ap1 >= 5).to_numpy() & (Xtr_raw[col(X, "Tuition fees up to date")] == 1).to_numpy() \
    & (Xtr_raw[col(X, "Debtor")] == 0).to_numpy()
err["FN_main_good_profile_train_oof"] = {"n": int((miss[MAIN] & good).sum()), "of_FN": int(miss[MAIN].sum())}

# ============================================================
# 10. HÌNH 7–15
# ============================================================

plt.rcParams.update({"font.size": 10, "figure.dpi": 110})
COL = {BASE: "#1f77b4", MAIN: "#d62728", ALT: "#2ca02c"}
LAB = {BASE: "Base: Logistic Regression", MAIN: "CatBoost + FE", ALT: "LightGBM + FE"}
prev_test = yb_test.mean()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / "hinh" / name, dpi=200)
    plt.close(fig)


fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for cfg in (BASE, MAIN, ALT):
    for a, k, t in zip(ax, ["AP", "F2_crossfit"], ["AP (Dropout)", "F2 tại ngưỡng cross-fit"]):
        a.plot(range(1, N_SPLITS + 1), fold_metric(cfg, k), "o-", color=COL[cfg], label=LAB[cfg])
        a.set_title(t)
        a.set_xlabel("Fold (CV trên Train)")
        a.set_xticks(range(1, N_SPLITS + 1))
ax[0].legend(fontsize=8)
save(fig, "hinh07_ap_f2_theo_fold.png")

fig, ax = plt.subplots(figsize=(7, 4))
p_main, p_base = oof[MAIN][:, dropout_id], oof[BASE][:, dropout_id]
for fn, name, ls in [(precision_score, "Precision", "-"), (recall_score, "Recall", "-")]:
    ax.plot(TAUS, [fn(yb_train, (p_main >= t).astype(int), zero_division=0) for t in TAUS], ls, label=f"{name} – CatBoost + FE")
ax.plot(TAUS, [f2_at(yb_train, p_main, t) for t in TAUS], "-", lw=2.2, color=COL[MAIN], label="F2 – CatBoost + FE")
ax.plot(TAUS, [f2_at(yb_train, p_base, t) for t in TAUS], "--", lw=2, color=COL[BASE], label="F2 – Base")
ax.axvline(tau_star[MAIN], color=COL[MAIN], ls=":", label=f"τ* CatBoost + FE = {tau_star[MAIN]:.2f}")
ax.axvline(tau_star[BASE], color=COL[BASE], ls=":", label=f"τ* Base = {tau_star[BASE]:.2f}")
ax.axvline(0.5, color="grey", ls="-.", label="ngưỡng 0,5")
ax.set_xlabel("Ngưỡng τ")
ax.set_ylabel("Giá trị")
ax.legend(fontsize=7, loc="lower left")
save(fig, "hinh08_f2_theo_nguong.png")

fig, ax = plt.subplots(figsize=(6, 4.6))
for cfg in (BASE, MAIN, ALT):
    p = test_proba[cfg][:, dropout_id]
    pr, rc, _ = precision_recall_curve(yb_test, p)
    ax.plot(rc, pr, color=COL[cfg], label=f"{LAB[cfg]} (AP = {average_precision_score(yb_test, p):.3f})")
    pred = (p >= tau_star[cfg]).astype(int)
    ax.plot(recall_score(yb_test, pred), precision_score(yb_test, pred), "o", ms=8, color=COL[cfg])
ax.axhline(prev_test, color="grey", ls="--", label=f"Tỷ lệ Dropout = {prev_test:.3f}")
ax.axvline(0.8, color="grey", ls=":", lw=1)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_ylim(0, 1.02)
ax.legend(fontsize=8, loc="lower left")
ax.set_title("Chấm tròn: điểm vận hành tại τ*", fontsize=9)
save(fig, "hinh09_pr_test.png")

fig, ax = plt.subplots(figsize=(5.2, 4.6))
for cfg in (BASE, MAIN, ALT):
    p = test_proba[cfg][:, dropout_id]
    fpr, tpr, _ = roc_curve(yb_test, p)
    ax.plot(fpr, tpr, color=COL[cfg], label=f"{LAB[cfg]} (AUC = {roc_auc_score(yb_test, p):.3f})")
ax.plot([0, 1], [0, 1], "--", color="grey", label="Ngẫu nhiên")
ax.set_xlabel("FPR (tỷ lệ cảnh báo nhầm)")
ax.set_ylabel("TPR (Recall)")
ax.legend(fontsize=8, loc="lower right")
save(fig, "hinh10_roc_test.png")

fig, ax = plt.subplots(figsize=(5.2, 4.6))
for cfg in (BASE, MAIN):
    p = test_proba[cfg][:, dropout_id]
    fr, mp = calibration_curve(yb_test, p, n_bins=10, strategy="quantile")
    ax.plot(mp, fr, "o-", color=COL[cfg], label=f"{LAB[cfg]} (Brier = {brier_score_loss(yb_test, p):.3f})")
ax.plot([0, 1], [0, 1], "--", color="grey", label="Lý tưởng")
ax.set_xlabel("Xác suất Dropout dự đoán (trung bình nhóm)")
ax.set_ylabel("Tỷ lệ Dropout thực tế")
ax.legend(fontsize=8, loc="upper left")
save(fig, "hinh11_calibration_test.png")

top = or_df.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(top.Feature.str.slice(0, 45), top.OR, color=["#d62728" if v > 1 else "#1f77b4" for v in top.OR])
ax.axvline(1, color="black", lw=1)
ax.set_xscale("log")
ax.set_xlabel("Odds ratio (thang log; biến số: mỗi 1 độ lệch chuẩn)")
save(fig, "hinh12_odds_ratio.png")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
pred_m = (test_proba[MAIN][:, dropout_id] >= tau_star[MAIN]).astype(int)
cm2 = confusion_matrix(yb_test, pred_m, labels=[0, 1])
cm3 = confusion_matrix(y_test, test_proba[MAIN].argmax(1), labels=list(range(n_classes)))
cm3n = cm3 / cm3.sum(1, keepdims=True)
for a, m_, lab, title, fmt in [
    (ax[0], cm2, ["Không Dropout", "Dropout"], f"Dropout so với còn lại, τ* = {tau_star[MAIN]:.2f}", "{:d}"),
    (ax[1], cm3n, class_names, "Ba lớp (argmax), chuẩn hóa theo hàng", "{:.2f}")]:
    a.imshow(m_, cmap="Blues")
    for i in range(m_.shape[0]):
        for j in range(m_.shape[1]):
            txt = fmt.format(m_[i, j]) if fmt == "{:d}" else f"{m_[i, j]:.2f}\n({cm3[i, j]})"
            a.text(j, i, txt, ha="center", va="center", color="black", fontsize=9)
    a.set_xticks(range(len(lab)))
    a.set_xticklabels(lab)
    a.set_yticks(range(len(lab)))
    a.set_yticklabels(lab)
    a.set_xlabel("Dự đoán")
    a.set_ylabel("Thực tế")
    a.set_title(title, fontsize=9)
save(fig, "hinh13_ma_tran_loi_test.png")

fig, ax = plt.subplots(figsize=(8, 4))
xx = np.arange(len(grp_df))
ax.bar(xx - 0.2, grp_df.miss_rate_Base, 0.4, color=COL[BASE], label="Base")
ax.bar(xx + 0.2, grp_df.miss_rate_CatBoost_FE, 0.4, color=COL[MAIN], label="CatBoost + FE")
ax.set_xticks(xx)
ax.set_xticklabels([f"{g}\n(n = {n})" for g, n in zip(grp_df.Group, grp_df.n_Dropout)], fontsize=7)
ax.set_ylabel("Tỷ lệ bỏ sót Dropout (tại τ*)")
ax.legend()
save(fig, "hinh14_bo_sot_theo_nhom.png")

fig, ax = plt.subplots(figsize=(7, 4))
p_d = oof[MAIN][yb_train == 1, dropout_id]
caught = p_d >= tau_star[MAIN]
bins = np.linspace(0, 1, 41)
ax.hist(p_d[caught], bins=bins, alpha=0.7, color=COL[MAIN], label=f"Bắt được ({caught.sum()})")
ax.hist(p_d[~caught], bins=bins, alpha=0.7, color="grey", label=f"Bỏ sót ({(~caught).sum()})")
ax.axvline(tau_star[MAIN], color="black", ls="--", label=f"τ* = {tau_star[MAIN]:.2f}")
ax.set_xlabel("p(Dropout) – CatBoost + FE, OOF trên Train")
ax.set_ylabel("Số sinh viên bỏ học thật")
ax.legend()
save(fig, "hinh15_phan_bo_p_dropout.png")

# ============================================================
# 11. GHI KẾT QUẢ
# ============================================================

pd.DataFrame({"index": idx_train, **{f"p_dropout__{cfg_name(c)}": oof[c][:, dropout_id] for c in configs},
              "y_true": y_train}).to_csv(OUT / "oof_train_predictions.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"index": idx_test, **{f"p_dropout__{cfg_name(c)}": test_proba[c][:, dropout_id] for c in configs},
              "y_true": y_test}).to_csv(OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
summary = {"n_train": int(len(idx_train)), "n_test": int(len(idx_test)), "dropout_test": int(yb_test.sum()),
           "tau_star": {cfg_name(c): tau_star[c] for c in configs}, "error_analysis": err}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

pd.set_option("display.width", 250)
print("\n== Ngưỡng ==\n", thr_df.round(4).to_string(index=False))
print("\n== Test ==\n", test_df.drop(columns=["TP", "FP", "FN", "TN"]).round(4).to_string(index=False))
print("\n== So với Base ==\n", pd.DataFrame(cmp_rows).round(4).to_string(index=False))
print("\n== Ablation FE ==\n", pd.DataFrame(abl_rows).round(4).to_string(index=False))
print("\n== Bỏ sót theo nhóm (OOF Train) ==\n", grp_df.round(3).to_string(index=False))
print("\n== Odds ratio (10 lớn nhất) ==\n", or_df.head(10).round(3).to_string(index=False))
print("\n", json.dumps(err, ensure_ascii=False, default=str))
print(f"\nĐã ghi kết quả vào {OUT}")
