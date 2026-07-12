# =========================
# Streamlit GUI for GBDT prediction
# Targets: NUT and NUT/NUO
# Function:
# 1. Train GBDT models only once
# 2. Save trained models locally
# 3. Load saved models directly next time
# 4. Chinese / English interface
# =========================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import optuna
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor


# =========================
# 1. Path settings
# =========================

base_dir = Path(__file__).resolve().parent
data_path = base_dir / "data.xlsx"

model_dir = base_dir / "saved_models"
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "GBDT_NUT_and_NUT_NUO_models.pkl"


# =========================
# 2. Basic settings
# =========================

RANDOM_STATE = 42
SHEET_NAME = "Sheet2"

TARGET_TASKS = {
    "NUT": "NUT",
    "NUT_NUO": "NUT/NUO"
}

BASE_FEATURE_COLS = [
    "D", "t", "D/t", "H", "H/D",
    "T", "TH", "TEI",
    "FY", "FC"
]

NUMERIC_COLS = BASE_FEATURE_COLS + ["NUT", "NUT/NUO"]


# =========================
# 3. Language dictionary
# =========================

TEXT = {
    "中文": {
        "title": "火灾后圆钢管混凝土柱承载力 GBDT 预测系统",
        "subtitle": "输入试件参数后，同时预测火灾后轴压承载力和承载力折损系数。",
        "language": "Language / 语言",
        "model_setting": "模型设置",
        "n_trials": "Optuna优化次数",
        "force_retrain": "重新训练并覆盖已保存模型",
        "training_info": "首次运行或重新训练时需要训练模型，请稍等。",
        "loading_model": "正在加载或训练 GBDT 模型...",
        "loaded_saved": "已加载本地保存模型，可直接预测。",
        "trained_saved": "模型训练完成，并已保存到本地。",
        "input": "输入参数",
        "geometry": "几何参数",
        "thermal": "热暴露参数",
        "material": "材料参数",
        "D": "D / mm",
        "t": "t / mm",
        "H": "H / mm",
        "T": "T / °C",
        "TH": "t_h / min",
        "TEI": "TEI",
        "FY": "f_y / MPa",
        "FC": "f_c / MPa",
        "CM": "冷却方式",
        "range_ok": "输入参数位于训练数据库范围内。",
        "range_warning": "注意：部分输入参数超出训练数据库范围，预测结果应谨慎使用。",
        "predict": "开始预测",
        "result": "预测结果",
        "nut": "火灾后轴压承载力 NUT",
        "nut_nuo": "承载力折损系数 NUT/NUO",
        "test_perf": "模型测试集性能",
        "best_params": "Best parameters / 最优参数",
        "data_error": "未找到 data.xlsx，请确认文件位于 C:\\Users\\zy\\Desktop\\ML\\GUI。",
        "delete_tip": "如果更换了 data.xlsx，请删除 saved_models 文件夹中的 pkl 文件后重新运行。",
    },
    "English": {
        "title": "GBDT Prediction System for Post-fire Circular CFST Columns",
        "subtitle": "Input specimen parameters to predict post-fire axial capacity and residual capacity ratio.",
        "language": "Language / 语言",
        "model_setting": "Model settings",
        "n_trials": "Optuna trials",
        "force_retrain": "Retrain and overwrite saved models",
        "training_info": "The first run or retraining requires model training. Please wait.",
        "loading_model": "Loading or training GBDT models...",
        "loaded_saved": "Saved local models loaded successfully.",
        "trained_saved": "Model training completed and saved locally.",
        "input": "Input parameters",
        "geometry": "Geometry",
        "thermal": "Thermal exposure",
        "material": "Material",
        "D": "D / mm",
        "t": "t / mm",
        "H": "H / mm",
        "T": "T / °C",
        "TH": "t_h / min",
        "TEI": "TEI",
        "FY": "f_y / MPa",
        "FC": "f_c / MPa",
        "CM": "Cooling method",
        "range_ok": "The input parameters are within the database range.",
        "range_warning": "Warning: some input parameters are outside the database range.",
        "predict": "Predict",
        "result": "Prediction results",
        "nut": "Post-fire axial capacity NUT",
        "nut_nuo": "Residual capacity ratio NUT/NUO",
        "test_perf": "Testing performance",
        "best_params": "Best parameters",
        "data_error": "data.xlsx was not found. Please check C:\\Users\\zy\\Desktop\\ML\\GUI.",
        "delete_tip": "If data.xlsx is changed, delete the pkl file in saved_models and rerun.",
    }
}


# =========================
# 4. Page configuration
# =========================

st.set_page_config(
    page_title="GBDT CFST Predictor",
    layout="wide"
)


# =========================
# 5. Load and preprocess data
# =========================

@st.cache_data
def load_data():

    if not data_path.exists():
        st.error(TEXT["中文"]["data_error"])
        st.stop()

    df = pd.read_excel(data_path, sheet_name=SHEET_NAME)
    df = df.dropna(how="all")

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["CM"] = df["CM"].astype(str).str.strip()

    required_cols = BASE_FEATURE_COLS + ["CM", "NUT", "NUT/NUO"]
    df = df.dropna(subset=required_cols).copy()

    df_model = pd.get_dummies(
        df,
        columns=["CM"],
        prefix="CM",
        drop_first=False
    )

    cm_cols = [col for col in df_model.columns if col.startswith("CM_")]
    feature_cols = BASE_FEATURE_COLS + cm_cols

    ranges = {}
    for col in BASE_FEATURE_COLS:
        ranges[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median())
        }

    cm_options = sorted(df["CM"].dropna().astype(str).unique().tolist())

    return df, df_model, feature_cols, cm_cols, cm_options, ranges


# =========================
# 6. Metrics
# =========================

def calc_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    nonzero = y_true != 0

    if np.sum(nonzero) > 0:
        mape = np.mean(
            np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])
        ) * 100
    else:
        mape = np.nan

    rrmse = rmse / np.mean(y_true) * 100

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE(%)": mape,
        "RRMSE(%)": rrmse
    }


# =========================
# 7. Build GBDT model
# =========================

def build_gbdt_model(trial):

    model = GradientBoostingRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 7),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        max_features=trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", None]
        ),
        loss="squared_error",
        random_state=RANDOM_STATE
    )

    return model


# =========================
# 8. Train one target
# =========================

def train_single_target(df_model, feature_cols, target_col, n_trials):

    required_cols = feature_cols + [target_col]
    data = df_model.dropna(subset=required_cols).copy()

    X = data[feature_cols].astype(float)
    y = data[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    def objective(trial):

        model = build_gbdt_model(trial)

        kf = KFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        rmse_scores = []

        for tr_idx, val_idx in kf.split(X_train):

            X_tr = X_train.iloc[tr_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train.iloc[tr_idx]
            y_val = y_train.iloc[val_idx]

            model.fit(X_tr, y_tr)
            y_val_pred = model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            rmse_scores.append(rmse)

        return np.mean(rmse_scores)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False
    )

    fixed_trial = optuna.trial.FixedTrial(study.best_params)
    model = build_gbdt_model(fixed_trial)

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = calc_metrics(y_train, y_train_pred)
    test_metrics = calc_metrics(y_test, y_test_pred)

    return {
        "model": model,
        "best_params": study.best_params,
        "best_cv_rmse": study.best_value,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "sample_count": len(data)
    }


# =========================
# 9. Train or load all models
# =========================

@st.cache_resource
def train_or_load_models(n_trials, force_retrain):

    df, df_model, feature_cols, cm_cols, cm_options, ranges = load_data()

    if model_path.exists() and not force_retrain:
        saved_data = joblib.load(model_path)
        saved_data["loaded_from_file"] = True
        return saved_data

    results = {}

    for target_key, target_col in TARGET_TASKS.items():
        results[target_key] = train_single_target(
            df_model=df_model,
            feature_cols=feature_cols,
            target_col=target_col,
            n_trials=n_trials
        )

    saved_data = {
        "models": results,
        "feature_cols": feature_cols,
        "cm_cols": cm_cols,
        "cm_options": cm_options,
        "ranges": ranges,
        "loaded_from_file": False
    }

    joblib.dump(saved_data, model_path)

    return saved_data


# =========================
# 10. Build input row
# =========================

def build_input_row(input_values, feature_cols, cm_cols):

    D = input_values["D"]
    t = input_values["t"]
    H = input_values["H"]

    row = {col: 0.0 for col in feature_cols}

    row["D"] = D
    row["t"] = t
    row["D/t"] = D / t if t != 0 else np.nan
    row["H"] = H
    row["H/D"] = H / D if D != 0 else np.nan
    row["T"] = input_values["T"]
    row["TH"] = input_values["TH"]
    row["TEI"] = input_values["TEI"]
    row["FY"] = input_values["FY"]
    row["FC"] = input_values["FC"]

    selected_cm = input_values["CM"]
    selected_cm_col = f"CM_{selected_cm}"

    for col in cm_cols:
        row[col] = 1.0 if col == selected_cm_col else 0.0

    return pd.DataFrame([row], columns=feature_cols).astype(float)


# =========================
# 11. Range check
# =========================

def check_input_range(input_values, ranges):

    check_cols = ["D", "t", "H", "T", "TH", "TEI", "FY", "FC"]

    out_cols = []

    for col in check_cols:
        value = input_values[col]
        low = ranges[col]["min"]
        high = ranges[col]["max"]

        if value < low or value > high:
            out_cols.append((col, value, low, high))

    D_t = input_values["D"] / input_values["t"]
    H_D = input_values["H"] / input_values["D"]

    if D_t < ranges["D/t"]["min"] or D_t > ranges["D/t"]["max"]:
        out_cols.append(("D/t", D_t, ranges["D/t"]["min"], ranges["D/t"]["max"]))

    if H_D < ranges["H/D"]["min"] or H_D > ranges["H/D"]["max"]:
        out_cols.append(("H/D", H_D, ranges["H/D"]["min"], ranges["H/D"]["max"]))

    return out_cols


# =========================
# 12. Sidebar
# =========================

language = st.sidebar.selectbox(
    "Language / 语言",
    ["中文", "English"],
    index=0
)

txt = TEXT[language]

st.sidebar.markdown(f"### {txt['model_setting']}")

n_trials = st.sidebar.slider(
    txt["n_trials"],
    min_value=20,
    max_value=200,
    value=100,
    step=10
)

force_retrain = st.sidebar.checkbox(
    txt["force_retrain"],
    value=False
)

st.sidebar.caption(txt["delete_tip"])


# =========================
# 13. Main page
# =========================

st.title(txt["title"])
st.caption(txt["subtitle"])

with st.spinner(txt["loading_model"]):
    model_bundle = train_or_load_models(
        n_trials=n_trials,
        force_retrain=force_retrain
    )

if model_bundle.get("loaded_from_file", False):
    st.success(txt["loaded_saved"])
else:
    st.success(txt["trained_saved"])

feature_cols = model_bundle["feature_cols"]
cm_cols = model_bundle["cm_cols"]
cm_options = model_bundle["cm_options"]
ranges = model_bundle["ranges"]
models = model_bundle["models"]


# =========================
# 14. Input panel
# =========================

st.markdown(f"## {txt['input']}")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"### {txt['geometry']}")

    D = st.number_input(
        txt["D"],
        value=float(ranges["D"]["median"]),
        min_value=0.0,
        step=1.0
    )

    t_val = st.number_input(
        txt["t"],
        value=float(ranges["t"]["median"]),
        min_value=0.01,
        step=0.1
    )

    H = st.number_input(
        txt["H"],
        value=float(ranges["H"]["median"]),
        min_value=0.0,
        step=1.0
    )

with col2:
    st.markdown(f"### {txt['thermal']}")

    T = st.number_input(
        txt["T"],
        value=float(ranges["T"]["median"]),
        min_value=0.0,
        step=10.0
    )

    TH = st.number_input(
        txt["TH"],
        value=float(ranges["TH"]["median"]),
        min_value=0.0,
        step=10.0
    )

    TEI_default = float(T * TH)

    TEI = st.number_input(
        txt["TEI"],
        value=TEI_default,
        min_value=0.0,
        step=100.0
    )

with col3:
    st.markdown(f"### {txt['material']}")

    FY = st.number_input(
        txt["FY"],
        value=float(ranges["FY"]["median"]),
        min_value=0.0,
        step=10.0
    )

    FC = st.number_input(
        txt["FC"],
        value=float(ranges["FC"]["median"]),
        min_value=0.0,
        step=1.0
    )

    CM = st.selectbox(
        txt["CM"],
        cm_options,
        index=0
    )


# =========================
# 15. Prediction
# =========================

input_values = {
    "D": D,
    "t": t_val,
    "H": H,
    "T": T,
    "TH": TH,
    "TEI": TEI,
    "FY": FY,
    "FC": FC,
    "CM": CM
}

out_cols = check_input_range(input_values, ranges)

if len(out_cols) == 0:
    st.info(txt["range_ok"])
else:
    st.warning(txt["range_warning"])

    warning_df = pd.DataFrame(
        out_cols,
        columns=["Variable", "Input value", "Database min", "Database max"]
    )

    st.dataframe(warning_df, use_container_width=True)

if st.button(txt["predict"], type="primary"):

    X_input = build_input_row(
        input_values=input_values,
        feature_cols=feature_cols,
        cm_cols=cm_cols
    )

    pred_nut = models["NUT"]["model"].predict(X_input)[0]
    pred_nut_nuo = models["NUT_NUO"]["model"].predict(X_input)[0]

    st.markdown(f"## {txt['result']}")

    r1, r2 = st.columns(2)

    with r1:
        st.metric(
            txt["nut"],
            f"{pred_nut:.2f} kN"
        )

    with r2:
        st.metric(
            txt["nut_nuo"],
            f"{pred_nut_nuo:.4f}"
        )


# =========================
# 16. Model information
# =========================

with st.expander(txt["test_perf"]):

    perf_rows = []

    for target_key, item in models.items():
        row = {
            "Target": TARGET_TASKS[target_key],
            "Sample count": item["sample_count"],
            "Best CV RMSE": item["best_cv_rmse"]
        }

        for key, value in item["test_metrics"].items():
            row[key] = value

        perf_rows.append(row)

    perf_df = pd.DataFrame(perf_rows)
    st.dataframe(perf_df, use_container_width=True)

with st.expander(txt["best_params"]):

    params_rows = []

    for target_key, item in models.items():
        row = {
            "Target": TARGET_TASKS[target_key]
        }

        row.update(item["best_params"])
        params_rows.append(row)

    params_df = pd.DataFrame(params_rows)
    st.dataframe(params_df, use_container_width=True)