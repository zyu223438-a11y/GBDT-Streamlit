"""Streamlit interface for the two saved publication-grouped GBDT models."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from preprocessing import input_frame, out_of_range, multivariate_distance


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
        "uncertainty": "预测不确定性与模型误差",
        "ci_note": "下表是固定测试指标在4篇测试文献上进行2000次文献级重采样得到的95%区间；它们不是单个新试件的预测区间。",
        "equation_check": "分析方程的独立估计",
        "equation_note": "下述90%经验预测区间由训练文献的分组交叉验证残差校准，并在76个固定测试试件上检验。区间较宽，仅供误差量级参考；它不属于GBDT预测值的区间。",
        "ood_warning": "输入参数的组合与训练文献中的试件差异较大。虽然部分单变量可能仍在范围内，模型对这一组合的可靠性尚未验证。",
        "screening_only": "仅供科研和工程初步筛查。不得直接用于结构设计、火灾后安全判定或替代专业检测；对新材料、构造与热历程需要独立验证。",
        "version": "模型及预处理版本",
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
        "uncertainty": "Prediction uncertainty and model error",
        "ci_note": "The 95% intervals below describe fixed-test metrics from 2,000 publication-level bootstrap draws over four test studies; they are not intervals for a new specimen prediction.",
        "equation_check": "Independent analytical-equation estimate",
        "equation_note": "The empirical 90% prediction interval below was calibrated from publication-grouped cross-validation residuals in the development set and checked on the 76 fixed-test specimens. It is broad and describes the equation estimate, not the GBDT prediction.",
        "ood_warning": "This combination of inputs is unlike the specimens in the development publications. Some individual inputs may still be within their observed ranges; reliability for this combination is unverified.",
        "screening_only": "For research and preliminary engineering screening only. Do not use directly for structural design or post-fire safety decisions, or instead of professional inspection. New materials, details and thermal histories require independent validation.",
        "version": "Model and preprocessing version",
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


def equation_prediction(values, coefficients):
    d, ts = values["D"], values["ts"]
    steel_area = np.pi * (d * ts - ts ** 2)
    concrete_area = np.pi / 4 * (d - 2 * ts) ** 2
    xt = max((values["Tmax"] - 20) / 1000, 0)
    xe = values["TEI"] / 100000
    base = (coefficients["a_s"] * values["FY"] * steel_area
            + coefficients["a_c"] * values["FC"] * concrete_area) / 1000
    reduction = np.exp(-coefficients["b_T"] * xt ** coefficients["p"]
                       - coefficients["b_E"] * xe ** coefficients["q"])
    return float(base * reduction)


st.set_page_config(page_title="CFST post-fire GBDT predictor", layout="wide")
language = st.sidebar.selectbox("Language / 语言", ["中文", "English"])
txt = LABELS[language]
st.title(txt["title"])
st.caption(txt["intro"])
st.warning(txt["screening_only"])

try:
    metadata, models = load_artifacts()
except FileNotFoundError:
    st.error(txt["missing"])
    st.stop()
except Exception as exc:
    st.error(f"{txt['integrity_error']} ({type(exc).__name__})")
    st.stop()

st.sidebar.caption(
    f"{txt['version']}: {metadata['release']} · "
    f"GBDT {metadata['model_run']} · {metadata['preprocessing_version']}"
)

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

novelty_score = multivariate_distance(values, metadata["multivariate_ood"])
if novelty_score > metadata["multivariate_ood"]["warning_cutoff"]:
    st.warning(txt["ood_warning"])

if st.button(txt["run"], type="primary"):
    prediction = {target: float(model.predict(frame)[0]) for target, model in models.items()}
    st.subheader(txt["result"])
    a, b = st.columns(2)
    a.metric(txt["capacity"], f"{prediction['NUT']:,.2f} kN")
    b.metric(txt["ratio"], f"{prediction['NUT/NUO']:.4f}")

    with st.expander(txt["equation_check"]):
        info = metadata["analytical_equation"]
        eq_value = equation_prediction(values, info["coefficients"])
        lo = eq_value * info["empirical_90pct_lower_multiplier"]
        hi = eq_value * info["empirical_90pct_upper_multiplier"]
        st.write(f"{eq_value:,.2f} kN; empirical 90% interval: {lo:,.2f}–{hi:,.2f} kN")
        st.caption(txt["equation_note"])

with st.expander(txt["metrics"]):
    st.caption(txt["metric_note"])
    st.dataframe(pd.DataFrame(metadata["test_metrics"]), hide_index=True)
    st.subheader(txt["uncertainty"])
    st.caption(txt["ci_note"])
    st.dataframe(pd.DataFrame(metadata["test_metric_95pct_cis"]), hide_index=True)
