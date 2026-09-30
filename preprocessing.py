"""Versioned preprocessing and input-domain checks for CFST GBDT models.

The 13-column order must match run 20260927_105305_848265 exactly.
"""

import numpy as np
import pandas as pd


EXPECTED_FEATURES = [
    "D", "ts", "D/ts", "H", "H/D", "Tmax", "th", "TEI",
    "FY", "FC", "CM_NC", "CM_UN", "CM_WC",
]


def input_frame(values, feature_columns):
    if list(feature_columns) != EXPECTED_FEATURES:
        raise ValueError("Feature order does not match the frozen preprocessing version")
    d = values["D"]
    ts = values["ts"]
    h = values["H"]
    if d <= 0 or ts <= 0 or d <= 2 * ts:
        raise ValueError("Physically invalid circular hollow section geometry")
    if values["CM"] not in ("NC", "UN", "WC"):
        raise ValueError("Unknown cooling method")
    row = {
        "D": d, "ts": ts, "D/ts": d / ts, "H": h, "H/D": h / d,
        "Tmax": values["Tmax"], "th": values["th"], "TEI": values["TEI"],
        "FY": values["FY"], "FC": values["FC"],
        "CM_NC": float(values["CM"] == "NC"),
        "CM_UN": float(values["CM"] == "UN"),
        "CM_WC": float(values["CM"] == "WC"),
    }
    frame = pd.DataFrame([row], columns=EXPECTED_FEATURES, dtype=float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Nonfinite model input")
    return frame


def out_of_range(frame, ranges):
    findings = []
    for col, bounds in ranges.items():
        value = float(frame.iloc[0][col])
        if value < bounds["min"] or value > bounds["max"]:
            findings.append((col, value, bounds["min"], bounds["max"]))
    return findings


def multivariate_distance(values, profile):
    variables = profile["variables"]
    sample = np.array([values[name] for name in variables], dtype=float)
    reference = np.array(profile["reference_values"], dtype=float)
    scales = np.array(profile["scale_iqr_or_range"], dtype=float)
    differences = np.sum(((reference - sample) / scales) ** 2, axis=1)
    classes = np.array(profile["reference_cooling"])
    differences += profile["cooling_mismatch_penalty_squared"] * (classes != values["CM"])
    return float(np.sqrt(np.min(differences)))
