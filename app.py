"""Streamlit interface for the two saved publication-grouped GBDT models."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
META_PATH = ROOT / "model_metadata.json"
MODEL_PATHS = {
    "NUT": ROOT / "models" / "NUT_GBDT_fixed_study_test_20260927_105305_848265.joblib",
    "NUT/NUO": ROOT / "models" / "NUT_RATIO_GBDT_fixed_study_test_20260927_105305_848265.joblib",
}

LABELS = {
    "中文": {
        "title": "圆形钢管混凝土柱火灾后承载力预测",
        "intro": "输入试件和热暴露参数，使用已训练的 GBDT 模型预测承载力及其比值。",
        "geometry": "几何参数",
        "thermal": "热暴露参数",
        "material": "材料与冷却方式",
        "run": "开始预测",
        "result": "预测结果",
        "capacity": "火灾后轴压承载力 Nᵤ,ₜ",
        "ratio": "承载力比值 Nᵤ,ₜ/Nᵤ,₀",
        "range_warning": "以下变量超出模型开发集的观测范围，预测结果需谨慎解释：",
        "range_ok": "输入参数均位于模型开发集的单变量观测范围内。",
        "tei_help": "TEI 应依据该试件的温度—时间曲线积分后输入，单位为 °C·min。",
        "tei_note": "TEI 由温度—时间曲线积分得到；界面不会用最高温度 × 持续时间代替。",
        "physical_error": "请检查几何参数：需要 D > 2tₛ、D > 0 且 tₛ > 0。",
        "integrity_error": "模型文件或输入列与当前版本不一致，请核对部署文件。",
        "metrics": "固定文献测试集上的 GBDT 指标",
        "metric_note": "指标对应 4 篇未参与开发的文献，共 76 个试件。",
        "missing": "缺少模型或元数据文件，请同时上传 app.py、model_metadata.json 和 models 文件夹。",
    },
    "English": {
        "title": "Post-fire capacity of circular CFST columns",
        "intro": "Enter specimen and thermal-exposure parameters to obtain predictions from the saved GBDT models.",
        "geometry": "Geometry",
        "thermal": "Thermal exposure",
        "material": "Materials and cooling",
        "run": "Predict",
        "result": "Predictions",
        "capacity": "Post-fire axial capacity Nᵤ,ₜ",
        "ratio": "Capacity ratio Nᵤ,ₜ/Nᵤ,₀",
        "range_warning": "The following inputs are outside the observed development-set range; interpret predictions cautiously:",
        "range_ok": "All inputs are within the observed univariate development-set ranges.",
        "tei_help": "Enter TEI calculated from the specimen's temperature–time curve, in °C·min.",
        "tei_note": "TEI is calculated by integrating the temperature–time curve; peak temperature × duration is not substituted.",
        "physical_error": "Check geometry: D > 2tₛ, D > 0 and tₛ > 0 are required.",
        "integrity_error": "The model files or input columns do not match this release. Check deployment files.",
        "metrics": "GBDT metrics on the fixed publication-level test set",
        "metric_note": "The metrics refer to 76 specimens from four publications excluded from model development.",
        "missing": "Required model or metadata files are missing. Upload app.py, model_metadata.json and the models folder together.",
    },
}


@st.cache_resource
def load_artifacts():
    if not META_PATH.is_file() or any(not path.is_file() for path in MODEL_PATHS.values()):
        raise FileNotFoundError("Missing deployment artifacts")
    with META_PATH.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    models = {target: joblib.load(path) for target, path in MODEL_PATHS.items()}
    expected = list(meta["feature_columns"])
    for model in models.values():
        if hasattr(model, "feature_names_in_") and list(model.feature_names_in_) != expected:
            raise ValueError("Model feature order does not match metadata")
    return meta, models


def input_frame(values, feature_columns):
    d = values["D"]
    ts = values["ts"]
    h = values["H"]
    row = {
        "D": d,
        "ts": ts,
        "D/ts": d / ts,
        "H": h,
        "H/D": h / d,
        "Tmax": values["Tmax"],
        "th": values["th"],
        "TEI": values["TEI"],
        "FY": values["FY"],
        "FC": values["FC"],
        "CM_NC": float(values["CM"] == "NC"),
        "CM_UN": float(values["CM"] == "UN"),
        "CM_WC": float(values["CM"] == "WC"),
    }
    if set(row) != set(feature_columns):
        raise ValueError("Feature names do not match training features")
    return pd.DataFrame([row], columns=feature_columns, dtype=float)


def out_of_range(frame, ranges):
    findings = []
    for col, bounds in ranges.items():
        value = float(frame.iloc[0][col])
        if not np.isfinite(value) or not bounds["min"] <= value <= bounds["max"]:
            findings.append((col, value, bounds["min"], bounds["max"]))
    return findings


st.set_page_config(page_title="CFST post-fire GBDT predictor", layout="wide")
language = st.sidebar.selectbox("Language / 语言", ["中文", "English"])
txt = LABELS[language]
st.title(txt["title"])
st.caption(txt["intro"])

try:
    metadata, models = load_artifacts()
except FileNotFoundError:
    st.error(txt["missing"])
    st.stop()
except Exception as exc:
    st.error(f"{txt['integrity_error']} ({type(exc).__name__})")
    st.stop()

defaults = metadata["defaults"]
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader(txt["geometry"])
    d = st.number_input("D (mm)", min_value=0.01, value=defaults["D"], step=1.0)
    ts = st.number_input("tₛ (mm)", min_value=0.01, value=defaults["ts"], step=0.1)
    h = st.number_input("H (mm)", min_value=0.01, value=defaults["H"], step=1.0)
with col2:
    st.subheader(txt["thermal"])
    tmax = st.number_input("Tₘₐₓ (°C)", min_value=0.0, value=defaults["Tmax"], step=10.0)
    th = st.number_input("tₕ (min)", min_value=0.0, value=defaults["th"], step=1.0)
    tei = st.number_input("TEI (°C·min)", min_value=0.0, value=defaults["TEI"], step=100.0, help=txt["tei_help"])
    st.caption(txt["tei_note"])
with col3:
    st.subheader(txt["material"])
    fy = st.number_input("fᵧ (MPa)", min_value=0.01, value=defaults["FY"], step=1.0)
    fc = st.number_input("f꜀ (MPa)", min_value=0.01, value=defaults["FC"], step=1.0)
    cm = st.selectbox("Cooling method / 冷却方式", metadata["cooling_methods"], index=metadata["cooling_methods"].index(metadata["default_cooling_method"]))

if d <= 2 * ts:
    st.error(txt["physical_error"])
    st.stop()

values = {"D": d, "ts": ts, "H": h, "Tmax": tmax, "th": th, "TEI": tei, "FY": fy, "FC": fc, "CM": cm}
try:
    frame = input_frame(values, metadata["feature_columns"])
except ValueError:
    st.error(txt["integrity_error"])
    st.stop()

findings = out_of_range(frame, metadata["development_ranges"])
if findings:
    st.warning(txt["range_warning"])
    st.dataframe(pd.DataFrame(findings, columns=["Variable", "Input", "Observed min", "Observed max"]), hide_index=True)
else:
    st.info(txt["range_ok"])

if st.button(txt["run"], type="primary"):
    prediction = {target: float(model.predict(frame)[0]) for target, model in models.items()}
    st.subheader(txt["result"])
    a, b = st.columns(2)
    a.metric(txt["capacity"], f"{prediction['NUT']:,.2f} kN")
    b.metric(txt["ratio"], f"{prediction['NUT/NUO']:.4f}")

with st.expander(txt["metrics"]):
    st.caption(txt["metric_note"])
    st.dataframe(pd.DataFrame(metadata["test_metrics"]), hide_index=True)
