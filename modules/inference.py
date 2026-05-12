"""
ESA CDSS Inference Engine
=========================
Core prediction engine that integrates:
  - XGBoost risk prediction model
  - K-Means phenotype classification
  - SHAP-based explainability
  - Patient-level clinical action recommendations
"""

import os
import json
import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    MODEL_PATH, CLUSTER_META_PATH, SUBTYPE_NAMES, SUBTYPE_SHORT_NAMES,
    SUBTYPE_DESCRIPTIONS, SUBTYPE_CHECKLISTS, FEATURE_LABELS, SHAP_HIDE,
    RISK_THRESHOLDS, RISK_COLORS, RISK_DESCRIPTIONS,
)
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fval(v) -> float:
    """Convert value to float, returning NaN on failure."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# Asset loading (version-independent: no sklearn pickle dependency)
# ---------------------------------------------------------------------------

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
XGB_MODEL_PATH = ASSETS_DIR / "xgb_model.json"
PREPROCESSOR_CONFIG_PATH = ASSETS_DIR / "preprocessor_config.json"


class CDSSAssets:
    """Lazy-loading singleton for all model assets."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if self._loaded:
            return

        # --- Load XGBoost model (native JSON format, sklearn-independent) ---
        if not XGB_MODEL_PATH.exists():
            raise FileNotFoundError(f"XGBoost model not found: {XGB_MODEL_PATH}")
        self.xgb_model = XGBClassifier()
        self.xgb_model.load_model(str(XGB_MODEL_PATH))

        # --- Load preprocessor config (JSON, no sklearn pickle) ---
        if not PREPROCESSOR_CONFIG_PATH.exists():
            raise FileNotFoundError(f"Preprocessor config not found: {PREPROCESSOR_CONFIG_PATH}")
        with open(PREPROCESSOR_CONFIG_PATH, "r", encoding="utf-8") as f:
            preproc_cfg = json.load(f)
        self.feat_names_out = preproc_cfg.get("feature_names_out")
        self._preproc_cfg = preproc_cfg

        # --- Load cluster metadata ---
        if not CLUSTER_META_PATH.exists():
            raise FileNotFoundError(f"Cluster metadata not found: {CLUSTER_META_PATH}")
        with open(CLUSTER_META_PATH, "r", encoding="utf-8") as f:
            self.cluster_meta = json.load(f)
        self._enrich_cluster_meta()

        # --- SHAP explainer ---
        self.explainer = shap.TreeExplainer(self.xgb_model)

        self._loaded = True

    def transform(self, df):
        """Apply preprocessing manually (sklearn-independent)."""
        cfg = self._preproc_cfg
        num_parts = {}
        cat_parts = {}

        for t_info in cfg["transformers"]:
            name = t_info["name"]
            columns = t_info["columns"]
            steps = t_info["steps"]

            if name == "num":
                step0 = steps[0].get("imputer" if "imputer" in steps[0] else list(steps[0].keys())[0],
                                     steps[0][list(steps[0].keys())[0]])
                step1 = steps[1].get("scaler" if "scaler" in steps[1] else list(steps[1].keys())[0],
                                     steps[1][list(steps[1].keys())[0]])

                fill_values = step0["statistics_"]
                means = step1["mean_"]
                scales = step1["scale_"]

                for i, col in enumerate(columns):
                    series = df[col].apply(pd.to_numeric, errors="coerce")
                    series = series.fillna(fill_values[i])
                    num_parts[f"num__{col}"] = (series - means[i]) / scales[i]

            elif name == "cat":
                step0 = steps[0].get("imputer" if "imputer" in steps[0] else list(steps[0].keys())[0],
                                     steps[0][list(steps[0].keys())[0]])
                step1 = steps[1].get("encoder" if "encoder" in steps[1] else list(steps[1].keys())[0],
                                     steps[1][list(steps[1].keys())[0]])

                categories = step1["categories_"]

                for i, col in enumerate(columns):
                    series = df[col].astype(str).fillna("")
                    for cat_val in categories[i]:
                        cat_parts[f"cat__{col}_{cat_val}"] = (series == cat_val).astype(float)

        all_parts = {**num_parts, **cat_parts}
        result = pd.DataFrame(all_parts, index=df.index)

        # Reorder to match training feature names
        if self.feat_names_out:
            for col in self.feat_names_out:
                if col not in result.columns:
                    result[col] = 0.0
            result = result[self.feat_names_out]

        return result.values

    def _enrich_cluster_meta(self):
        """Compute imputation values and scaler params if missing."""
        meta = self.cluster_meta
        if {"imputation_values", "scaler_mean", "scaler_scale"}.issubset(meta):
            return

        cluster_dir = CLUSTER_META_PATH.parent
        for name in ("phenotype_analysis_dataset_revised.csv",
                     "phenotype_analysis_dataset.csv"):
            p = cluster_dir / name
            if p.exists():
                break
        else:
            return

        features = meta["cluster_features"]
        df = pd.read_csv(p, usecols=features).apply(pd.to_numeric, errors="coerce")
        imp = df.median(numeric_only=True).fillna(0)
        filled = df.fillna(imp).fillna(0)
        scale = filled.std(axis=0, ddof=0).replace(0, 1)

        meta["imputation_values"] = imp.to_dict()
        meta["scaler_mean"] = filled.mean(axis=0).to_dict()
        meta["scaler_scale"] = scale.to_dict()


def get_assets() -> CDSSAssets:
    """Get or initialize model assets."""
    assets = CDSSAssets()
    assets.load()
    return assets


# ---------------------------------------------------------------------------
# Input processing
# ---------------------------------------------------------------------------

def build_input_dataframe(values: Dict[str, Any]) -> pd.DataFrame:
    """Build a complete feature DataFrame from user input values."""
    row = values.copy()

    # Fill pipeline-required categorical/identifier fields
    row.setdefault("center_creator", "\u6cb3\u6e90\u5e02\u7d2b\u91d1\u53bf\u4e2d\u533b\u9662\u8840\u900f\u4e2d\u5fc3")
    row.setdefault("receiving_center", "\u4e0d\u8be6")
    row.setdefault("sex", "\u7537")
    row.setdefault("patient_status", "\u5728\u900f")
    row.setdefault("primary_disease", "\u539f\u53d1\u75c5\u4e0d\u660e\u786e")
    row.setdefault("esa_use", "\u4f7f\u7528")
    row.setdefault("esa_type", "\u91cd\u7ec4\u4eba\u4fc3\u7ea2\u7ec6\u80de\u751f\u6210\u7d20\uff08rHuEPO\uff09")
    row.setdefault("esa_unit", "IU")
    iron_flag = _fval(row.get("iron_use_flag", 0))
    hif_flag = _fval(row.get("hif_use_flag", 0))
    row.setdefault("iron_use", "\u4f7f\u7528" if np.isfinite(iron_flag) and int(iron_flag) == 1 else "\u672a\u4f7f\u7528")
    row.setdefault("hif_use", "\u4f7f\u7528" if np.isfinite(hif_flag) and int(hif_flag) == 1 else "\u672a\u4f7f\u7528")

    route = row.get("esa_route", "Subcutaneous")
    route_text = str(route)
    if route_text in ("Subcutaneous", "SC", "subcutaneous"):
        row["esa_route"] = "\u76ae\u4e0b"
    elif route_text in ("Intravenous", "IV", "intravenous"):
        row["esa_route"] = "\u9759\u8109"
    # Compute derived features
    route = row.get("esa_route", "Subcutaneous")
    dose = _fval(row.get("esa_dose", 0.0))
    eq_dose = dose if route in ("Subcutaneous", "\u76ae\u4e0b") else dose * 2 / 3
    row.setdefault("eq_esa_dose", eq_dose)

    hb = _fval(row.get("hb"))
    wt = _fval(row.get("dry_weight"))
    if np.isfinite(hb) and np.isfinite(wt) and hb > 0 and wt > 0:
        row.setdefault("eri", eq_dose / wt / (hb / 10.0))
    else:
        row.setdefault("eri", np.nan)

    ferritin = _fval(row.get("ferritin_mean", row.get("ferritin_latest", row.get("ferritin"))))
    tsat = _fval(row.get("tsat_mean", row.get("tsat_latest", row.get("tsat"))))
    crp = _fval(row.get("crp"))

    if np.isfinite(ferritin):
        row.setdefault("ferritin", ferritin)
        row.setdefault("ferritin_mean", ferritin)
        row.setdefault("ferritin_latest", ferritin)
        row.setdefault("log_ferritin", np.log1p(max(ferritin, 0.0)))
        row.setdefault("ferritin_deficiency", 1 if ferritin < 200 else 0)
        row.setdefault("ferritin_missing", 0)
    else:
        row.setdefault("ferritin_mean", np.nan)
        row.setdefault("ferritin_latest", np.nan)
        row.setdefault("log_ferritin", np.nan)
        row.setdefault("ferritin_deficiency", 0)
        row.setdefault("ferritin_missing", 1)

    if np.isfinite(tsat):
        row.setdefault("tsat", tsat)
        row.setdefault("tsat_mean", tsat)
        row.setdefault("tsat_latest", tsat)
        row.setdefault("tsat_deficiency", 1 if tsat < 20 else 0)
        row.setdefault("tsat_missing", 0)
    else:
        row.setdefault("tsat_mean", np.nan)
        row.setdefault("tsat_latest", np.nan)
        row.setdefault("tsat_deficiency", 0)
        row.setdefault("tsat_missing", 1)

    log_ferritin = _fval(row.get("log_ferritin"))
    if np.isfinite(log_ferritin) and np.isfinite(crp):
        row.setdefault("ferritin_crp_interaction", log_ferritin * crp)
    else:
        row.setdefault("ferritin_crp_interaction", np.nan)

    row.setdefault("prev_ferritin_mean", row.get("ferritin_mean", np.nan))
    row.setdefault("prev_tsat_mean", row.get("tsat_mean", np.nan))
    row.setdefault("delta_ferritin_mean", 0)
    row.setdefault("delta_tsat_mean", 0)

    row.setdefault("esa_use_flag", 1)
    row.setdefault("prior_low_response_proxy", 0)

    # Map current-quarter BP/IDH to legacy column names
    row.setdefault("current_pre_sbp_mean", row.get("pre_sbp_q1_mean", np.nan))
    row.setdefault("current_pre_dbp_mean", row.get("pre_dbp_q1_mean", np.nan))
    row.setdefault("current_idh_any", row.get("idh_any_q1", 0))
    row.setdefault("pre_sbp_q1_mean", row.get("current_pre_sbp_mean", 0))
    row.setdefault("pre_dbp_q1_mean", row.get("current_pre_dbp_mean", 0))
    row.setdefault("pre_sbp_q1_std", 0)
    row.setdefault("pre_dbp_q1_std", 0)
    row.setdefault("idh_any_q1", row.get("current_idh_any", 0))
    row.setdefault("idh_count_q1", row.get("current_idh_any", 0))

    for col in (
        "delta_hb", "delta_esa_dose", "delta_eri", "delta_crp",
        "delta_albumin", "delta_ktv", "delta_pth",
        "pre_sbp_q2_mean", "pre_dbp_q2_mean", "idh_any_q2",
        "pre_sbp_q3_mean", "pre_dbp_q3_mean", "idh_any_q3",
    ):
        row.setdefault(col, 0)

    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Risk stratification
# ---------------------------------------------------------------------------

def assign_risk_level(prob: float) -> str:
    """Assign risk level based on predicted probability."""
    if prob <= 0.20:
        return "Low"
    if prob <= 0.50:
        return "Intermediate"
    if prob <= 0.80:
        return "High"
    return "Very High"


# ---------------------------------------------------------------------------
# Phenotype classification
# ---------------------------------------------------------------------------

def assign_phenotype(values: Dict[str, Any], cluster_meta: Dict) -> Dict:
    """Classify patient into a phenotype cluster."""
    centroids = np.array(
        cluster_meta.get("centroids_scaled", cluster_meta.get("centroids")),
        dtype=float,
    )
    feature_order = cluster_meta["cluster_features"]
    imp = cluster_meta.get("imputation_values", {})
    mean = cluster_meta.get("scaler_mean", {})
    scale = cluster_meta.get("scaler_scale", {})

    prepared = values.copy()
    crp = _fval(prepared.get("crp"))
    cap = _fval(cluster_meta.get("crp_p99_cap"))
    if np.isfinite(crp):
        capped = min(max(crp, 0.0), cap) if np.isfinite(cap) else max(crp, 0.0)
        prepared["crp_winsor99"] = capped
        prepared["log_crp_w99"] = np.log1p(capped)
    prepared.setdefault("current_pre_sbp_mean", prepared.get("pre_sbp_q1_mean", np.nan))
    prepared.setdefault("current_pre_dbp_mean", prepared.get("pre_dbp_q1_mean", np.nan))
    prepared.setdefault("current_idh_any", prepared.get("idh_any_q1", 0))

    arr = []
    for col in feature_order:
        v = _fval(prepared.get(col))
        if not np.isfinite(v):
            v = _fval(imp.get(col, 0))
        if mean and scale:
            s = _fval(scale.get(col, 1))
            v = (v - _fval(mean.get(col, 0))) / (s if s else 1)
        arr.append(v)

    vec = np.array(arr, dtype=float)
    dists = np.linalg.norm(centroids - vec, axis=1)
    exp_neg = np.exp(-dists)
    probs = exp_neg / exp_neg.sum()

    idx = int(np.argmin(dists))
    name_map = SUBTYPE_NAMES

    distances = {}
    for i, name in name_map.items():
        distances[name] = {
            "distance": round(float(dists[i]), 3),
            "similarity": round(float(probs[i]), 3),
        }

    return {
        "assigned_index": idx,
        "assigned": name_map[idx],
        "assigned_short": SUBTYPE_SHORT_NAMES[idx],
        "description": SUBTYPE_DESCRIPTIONS[idx],
        "checklist": SUBTYPE_CHECKLISTS[idx],
        "distances": distances,
    }


# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------

def compute_shap(assets: CDSSAssets, df: pd.DataFrame) -> Dict:
    """Compute SHAP values and extract risk drivers/protectors."""
    X_processed = assets.transform(df)
    X_processed = np.atleast_2d(X_processed)

    sv = assets.explainer.shap_values(X_processed)
    if isinstance(sv, list):
        sv = sv[1]
    sv = sv[0]

    fnames = (assets.feat_names_out
              if assets.feat_names_out
              else [f"f{i}" for i in range(len(sv))])

    def _strip(name):
        if name.startswith("num__"):
            return name[5:]
        if name.startswith("cat__"):
            return name[5:]
        return name

    contributions = []
    for i, fname in enumerate(fnames):
        bare = _strip(fname)
        if bare in SHAP_HIDE:
            continue
        if fname.startswith("cat__"):
            continue
        contributions.append({
            "feature": bare,
            "label": FEATURE_LABELS.get(bare, bare),
            "shap": float(sv[i]),
            "abs_shap": abs(float(sv[i])),
        })
    contributions.sort(key=lambda x: x["abs_shap"], reverse=True)

    drivers = [c for c in contributions if c["shap"] > 0][:10]
    protective = [c for c in contributions if c["shap"] < 0][:5]

    return {
        "contributions": contributions,
        "drivers": drivers,
        "protective": protective,
        "base_value": float(assets.explainer.expected_value),
    }


# ---------------------------------------------------------------------------
# Decision recommendation engine
# ---------------------------------------------------------------------------

TIMEFRAME_RANK = {
    "Immediate / same day": 0,
    "Within 1 week": 1,
    "2-4 weeks": 2,
    "4-8 week reassessment": 3,
}

SEVERITY_RANK = {
    "emergency": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}

TIER_RANK = {
    "urgent": 0,
    "primary": 1,
    "phenotype": 2,
    "supportive": 3,
}

CLINICAL_SAFETY_NOTE = (
    "Apply within local protocols, medication contraindications, and the responsible clinician's judgment."
)


def _domain(feat: str) -> str:
    if feat in ("ferritin", "ferritin_mean", "ferritin_latest", "log_ferritin",
                "tsat", "tsat_mean", "tsat_latest", "ferritin_deficiency",
                "tsat_deficiency", "ferritin_crp_interaction", "ferritin_missing",
                "tsat_missing", "delta_ferritin_mean", "delta_tsat_mean"):
        return "iron_status"
    if feat in ("esa_dose", "eq_esa_dose", "eri"):
        return "esa"
    if feat in ("current_pre_sbp_mean", "current_pre_dbp_mean"):
        return "bp"
    if feat in ("current_idh_any", "idh_any_q1", "idh_count_q1"):
        return "idh"
    if feat in ("ktv", "delta_ktv"):
        return "dialysis"
    if feat in ("urr",):
        return "dialysis"
    if feat in ("pth", "phosphorus", "calcium"):
        return "mbd"
    return feat


def _fmt(value, unit="", digits=1, missing="not available"):
    if not np.isfinite(value):
        return missing
    if digits == 0:
        text = f"{value:.0f}"
    elif digits == 2:
        text = f"{value:.2f}"
    else:
        text = f"{value:.1f}"
    return f"{text} {unit}".strip()


def _yes_no(flag: bool) -> str:
    return "present" if flag else "absent"


def _patient_context(values: Dict[str, Any]) -> Dict[str, Any]:
    hb = _fval(values.get("hb"))
    dry_weight = _fval(values.get("dry_weight"))
    esa_dose = _fval(values.get("esa_dose"))
    eq_dose = _fval(values.get("eq_esa_dose"))
    if not np.isfinite(eq_dose):
        route = str(values.get("esa_route", "Subcutaneous"))
        eq_dose = esa_dose if route in ("Subcutaneous", "SC", "subcutaneous", "\u76ae\u4e0b") else esa_dose * 2 / 3
    eri = _fval(values.get("eri"))
    if not np.isfinite(eri) and np.isfinite(eq_dose) and np.isfinite(dry_weight) and np.isfinite(hb) and dry_weight > 0 and hb > 0:
        eri = eq_dose / dry_weight / (hb / 10.0)

    crp = _fval(values.get("crp"))
    albumin = _fval(values.get("albumin"))
    ktv = _fval(values.get("ktv"))
    urr = _fval(values.get("urr"))
    pth = _fval(values.get("pth"))
    calcium = _fval(values.get("calcium"))
    phosphorus = _fval(values.get("phosphorus"))
    potassium = _fval(values.get("potassium"))
    sodium = _fval(values.get("sodium"))
    sbp = _fval(values.get("current_pre_sbp_mean", values.get("pre_sbp_q1_mean")))
    dbp = _fval(values.get("current_pre_dbp_mean", values.get("pre_dbp_q1_mean")))
    idh_raw = _fval(values.get("current_idh_any", values.get("idh_any_q1", 0)))
    idh = bool(np.isfinite(idh_raw) and int(idh_raw) == 1)
    ferritin = _fval(values.get("ferritin_mean", values.get("ferritin_latest", values.get("ferritin"))))
    tsat = _fval(values.get("tsat_mean", values.get("tsat_latest", values.get("tsat"))))
    ferritin_missing = bool(np.isfinite(_fval(values.get("ferritin_missing"))) and _fval(values.get("ferritin_missing")) == 1)
    tsat_missing = bool(np.isfinite(_fval(values.get("tsat_missing"))) and _fval(values.get("tsat_missing")) == 1)
    iron_raw = _fval(values.get("iron_use_flag", 0))
    hif_raw = _fval(values.get("hif_use_flag", 0))
    iron_use = bool(np.isfinite(iron_raw) and int(iron_raw) == 1)
    hif_use = bool(np.isfinite(hif_raw) and int(hif_raw) == 1)

    iron_pattern = "not classifiable"
    if ferritin_missing or tsat_missing or not np.isfinite(ferritin) or not np.isfinite(tsat):
        iron_pattern = "incomplete iron assessment"
    elif tsat < 20 and ferritin < 200:
        iron_pattern = "absolute iron deficiency"
    elif tsat < 20 and ferritin >= 200:
        iron_pattern = "functional iron restriction"
    elif ferritin < 200:
        iron_pattern = "depleted iron stores"
    elif ferritin > 800 and tsat < 25:
        iron_pattern = "high ferritin with limited circulating iron"
    else:
        iron_pattern = "no major iron restriction signal"

    labels = []
    if np.isfinite(hb) and hb < 100:
        labels.append(f"low Hb {_fmt(hb, 'g/L')}")
    if np.isfinite(eri) and eri > 12:
        labels.append(f"high ERI {_fmt(eri)}")
    if np.isfinite(crp) and crp > 5:
        labels.append(f"inflammation CRP {_fmt(crp, 'mg/L')}")
    if np.isfinite(albumin) and albumin < 35:
        labels.append(f"low albumin {_fmt(albumin, 'g/L')}")
    if np.isfinite(ktv) and ktv < 1.2:
        labels.append(f"low Kt/V {_fmt(ktv, digits=2)}")
    if np.isfinite(urr) and urr < 65:
        labels.append(f"low URR {_fmt(urr, '%')}")
    if np.isfinite(pth) and pth > 600:
        labels.append(f"marked PTH elevation {_fmt(pth, 'pg/mL', 0)}")
    elif np.isfinite(pth) and pth > 300:
        labels.append(f"PTH elevation {_fmt(pth, 'pg/mL', 0)}")
    if np.isfinite(phosphorus) and phosphorus > 1.78:
        labels.append(f"hyperphosphatemia {_fmt(phosphorus, 'mmol/L', 2)}")
    if idh:
        labels.append("intradialytic hypotension")

    return {
        "hb": hb,
        "dry_weight": dry_weight,
        "esa_dose": esa_dose,
        "eq_dose": eq_dose,
        "eri": eri,
        "crp": crp,
        "albumin": albumin,
        "ktv": ktv,
        "urr": urr,
        "pth": pth,
        "calcium": calcium,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "sodium": sodium,
        "sbp": sbp,
        "dbp": dbp,
        "idh": idh,
        "ferritin": ferritin,
        "tsat": tsat,
        "ferritin_missing": ferritin_missing,
        "tsat_missing": tsat_missing,
        "iron_use": iron_use,
        "hif_use": hif_use,
        "iron_pattern": iron_pattern,
        "labels": labels,
    }


def _sg(tier: str, severity: str, timeframe: str, title: str, rationale: str,
        actions: List[str], avoid: Optional[str] = None, monitoring: Optional[str] = None,
        detail: Optional[str] = None, feature: Optional[str] = None,
        evidence: str = "Clinical best practice", priority: int = 0,
        shap: Optional[float] = None) -> Dict[str, Any]:
    if detail is None:
        detail_parts = [rationale]
        if actions:
            detail_parts.append("Actions: " + "; ".join(actions))
        if avoid:
            detail_parts.append("Avoid/caution: " + avoid)
        if monitoring:
            detail_parts.append("Monitoring: " + monitoring)
        if detail_parts:
            detail_parts.append(CLINICAL_SAFETY_NOTE)
        detail = "\n".join(detail_parts)
    return {
        "tier": tier,
        "severity": severity,
        "timeframe": timeframe,
        "title": title,
        "detail": detail,
        "rationale": rationale,
        "actions": actions,
        "avoid": avoid or "None specific beyond standard contraindication review.",
        "monitoring": monitoring or "Reassess after the selected intervention interval.",
        "feature": feature,
        "evidence": evidence,
        "priority": priority,
        "shap": shap,
    }


def _driver_suggestion(feat, sv, values):
    ctx = _patient_context(values)
    tag = f"SHAP +{sv:.3f}"
    hb = ctx["hb"]
    eri = ctx["eri"]
    crp = ctx["crp"]
    alb = ctx["albumin"]
    ktv = ctx["ktv"]
    urr = ctx["urr"]
    pth = ctx["pth"]
    ca = ctx["calcium"]
    phos = ctx["phosphorus"]
    sbp = ctx["sbp"]
    dbp = ctx["dbp"]
    esa_dose = ctx["esa_dose"]
    ferritin = ctx["ferritin"]
    tsat = ctx["tsat"]

    if feat == "hb" and np.isfinite(hb) and hb < 110:
        return _sg(
            "primary", "medium", "2-4 weeks",
            f"Low hemoglobin requires cause-directed anemia review ({tag})",
            f"Hb is {_fmt(hb, 'g/L')}; the patient's iron pattern is {ctx['iron_pattern']}, CRP is {_fmt(crp, 'mg/L')}, and ERI is {_fmt(eri)}.",
            [
                "Confirm iron status with ferritin and TSAT if not current",
                "Check for occult blood loss, access bleeding, recent hospitalization, and hemolysis when clinically indicated",
                "Optimize inflammation, iron availability, dialysis adequacy, and CKD-MBD before medication intensification",
            ],
            "Do not increase ESA reflexively when inflammation, iron restriction, underdialysis, or uncontrolled MBD is present.",
            "Repeat Hb, ESA dose, ERI, CRP, ferritin, and TSAT in 2-4 weeks if high-risk drivers are active.",
            feature="hb", evidence="KDIGO anemia guidance; clinical workflow", shap=sv,
        )

    if feat in ("eri", "eq_esa_dose", "esa_dose"):
        if (feat == "eri" and np.isfinite(eri) and eri > 12) or (np.isfinite(esa_dose) and esa_dose > 12000):
            dose_text = _fmt(esa_dose, "IU/week", 0)
            return _sg(
                "primary", "high", "Within 1 week",
                f"Avoid reflexive ESA escalation; complete low-response workup ({tag})",
                f"ESA dose is {dose_text} and ERI is {_fmt(eri)}, suggesting possible ESA hyporesponsiveness rather than simple underdosing.",
                [
                    "Prioritize iron restriction, inflammation, dialysis adequacy, CKD-MBD, bleeding, and hemolysis review",
                    "Document the active reversible drivers before any ESA dose change",
                    "Use local anemia protocol for any modest dose adjustment after reversible drivers are addressed",
                ],
                "Avoid supratherapeutic ESA escalation while Hb remains low because cardiovascular and thrombotic risks rise with high ESA exposure.",
                "Track Hb, ESA dose, ERI, blood pressure, and adverse events over the next 2-4 weeks.",
                feature=feat, evidence="KDIGO anemia guidance; ESA safety communications", shap=sv,
            )

    if _domain(feat) == "iron_status":
        if ctx["ferritin_missing"] or ctx["tsat_missing"] or not np.isfinite(ferritin) or not np.isfinite(tsat):
            missing = []
            if ctx["ferritin_missing"] or not np.isfinite(ferritin):
                missing.append("ferritin")
            if ctx["tsat_missing"] or not np.isfinite(tsat):
                missing.append("TSAT")
            return _sg(
                "primary", "high", "Within 1 week",
                f"Complete iron assessment before anemia treatment changes ({tag})",
                f"Missing {', '.join(missing)} prevents distinction between absolute iron deficiency and functional iron restriction.",
                [
                    "Order ferritin and TSAT together",
                    "Review recent iron exposure, infection signs, blood loss, and transfusion history",
                    "Reclassify iron pattern before changing ESA or iron dose",
                ],
                "Avoid attributing low Hb to ESA resistance until iron availability is known.",
                "Recheck iron indices 4-8 weeks after any iron intervention.",
                feature=feat, evidence="KDIGO anemia guidance", shap=sv,
            )
        if np.isfinite(tsat) and tsat < 20 and np.isfinite(ferritin) and ferritin < 200:
            return _sg(
                "primary", "high", "Within 1 week",
                f"Absolute iron deficiency is a priority reversible driver ({tag})",
                f"TSAT is {_fmt(tsat, '%')} and ferritin is {_fmt(ferritin, 'ng/mL', 0)}, consistent with depleted iron stores.",
                [
                    "Assess recent blood loss, access bleeding, gastrointestinal symptoms, and iron adherence or dosing",
                    "Optimize iron repletion according to local hemodialysis anemia protocol",
                    "Reassess ESA response after iron availability improves",
                ],
                "Avoid increasing ESA before iron repletion unless there is a separate urgent indication.",
                "Repeat Hb, ferritin, and TSAT in 4-8 weeks after iron optimization.",
                feature=feat, evidence="KDIGO anemia guidance", shap=sv,
            )
        if np.isfinite(tsat) and tsat < 20 and np.isfinite(ferritin) and ferritin >= 200:
            severity = "high" if (np.isfinite(crp) and crp > 5) or ferritin > 800 else "medium"
            return _sg(
                "primary", severity, "Within 1 week" if severity == "high" else "2-4 weeks",
                f"Functional iron restriction needs inflammation-aware management ({tag})",
                f"TSAT is {_fmt(tsat, '%')} with ferritin {_fmt(ferritin, 'ng/mL', 0)} and CRP {_fmt(crp, 'mg/L')}, suggesting restricted circulating iron rather than simple depletion.",
                [
                    "Look for active infection or inflammatory disease, especially vascular access, respiratory, urinary, skin, and dental sources",
                    "Decide whether to continue, hold, or adjust iron using local protocol and infection status",
                    "Address inflammation before interpreting ESA failure",
                ],
                "Avoid adding iron or escalating ESA automatically when ferritin is high or active infection is suspected.",
                "Repeat CRP, ferritin, TSAT, Hb, and ERI in 2-4 weeks if inflammation is active; otherwise 4-8 weeks.",
                feature=feat, evidence="KDIGO anemia guidance; inflammation-mediated iron restriction literature", shap=sv,
            )
        if np.isfinite(ferritin) and ferritin < 200:
            return _sg(
                "primary", "medium", "2-4 weeks",
                f"Low ferritin suggests depleted iron stores ({tag})",
                f"Ferritin is {_fmt(ferritin, 'ng/mL', 0)} with TSAT {_fmt(tsat, '%')}; iron stores may be insufficient for erythropoiesis.",
                [
                    "Review iron prescription, adherence, and recent missed doses",
                    "Assess for chronic blood loss and access-related loss",
                    "Optimize iron therapy before increasing ESA dose",
                ],
                "Avoid labeling the patient ESA-resistant until iron stores are corrected.",
                "Repeat Hb, ferritin, and TSAT in 4-8 weeks.",
                feature=feat, evidence="KDIGO anemia guidance", shap=sv,
            )

    if feat == "crp" and np.isfinite(crp) and crp > 5:
        severity = "high" if crp > 50 else "medium"
        timeframe = "Immediate / same day" if crp > 50 else "Within 1 week"
        return _sg(
            "primary", severity, timeframe,
            f"Inflammation is likely limiting ESA response ({tag})",
            f"CRP is {_fmt(crp, 'mg/L')}; albumin is {_fmt(alb, 'g/L')} and iron pattern is {ctx['iron_pattern']}.",
            [
                "Screen for vascular access infection and recent fever, chills, hospitalization, wounds, respiratory, urinary, and dental symptoms",
                "Use cultures or imaging when clinical findings support infection workup",
                "Treat the inflammatory source before judging ESA failure",
            ],
            "Avoid treating an inflammatory ESA-resistant pattern with ESA dose escalation alone.",
            "Repeat CRP, Hb, ferritin, TSAT, and ERI in 2-4 weeks, sooner if clinically unstable.",
            feature="crp", evidence="KDIGO anemia guidance; inflammation-mediated ESA resistance literature", shap=sv,
        )

    if feat == "albumin" and np.isfinite(alb) and alb < 35:
        severity = "high" if alb < 30 else "medium"
        return _sg(
            "primary", severity, "Within 1 week" if severity == "high" else "2-4 weeks",
            f"Nutrition-inflammation risk is contributing to low response ({tag})",
            f"Albumin is {_fmt(alb, 'g/L')}; CRP is {_fmt(crp, 'mg/L')} and dialysis adequacy is Kt/V {_fmt(ktv, digits=2)} / URR {_fmt(urr, '%')}.",
            [
                "Assess dietary intake, protein catabolic rate if available, gastrointestinal losses, edema, inflammation, and hospitalization history",
                "Request dietitian-led nutrition intervention when albumin is persistently below target",
                "Coordinate inflammation control, nutrition support, and dialysis prescription review",
            ],
            "Avoid focusing only on Hb correction when severe hypoalbuminemia indicates higher competing clinical risk.",
            "Recheck albumin, CRP, dietary adherence, interdialytic weight gain, Hb, and ERI in 2-4 weeks.",
            feature="albumin", evidence="KDOQI Nutrition 2020; clinical best practice", shap=sv,
        )

    if feat in ("ktv", "delta_ktv", "urr") and ((np.isfinite(ktv) and ktv < 1.2) or (np.isfinite(urr) and urr < 65)):
        return _sg(
            "primary", "high", "Within 1 week",
            f"Delivered dialysis dose should be optimized before ESA escalation ({tag})",
            f"Kt/V is {_fmt(ktv, digits=2)} and URR is {_fmt(urr, '%')}; inadequate clearance can suppress erythropoiesis and worsen inflammation.",
            [
                "Check treatment time, shortened or missed sessions, blood flow, dialyzer performance, access recirculation, and needle placement",
                "Consider increasing treatment time or adjusting prescription if delivered adequacy remains below target",
                "Review whether IDH or low blood pressure is limiting delivered dialysis",
            ],
            "Avoid escalating anemia therapy without correcting underdialysis when Kt/V or URR is below minimum targets.",
            "Repeat Kt/V, URR, IDH frequency, Hb, and ERI after the next monthly adequacy assessment.",
            feature=feat, evidence="KDOQI hemodialysis adequacy guidance", shap=sv,
        )

    if feat in ("pth", "phosphorus", "calcium"):
        mbd_signal = ((np.isfinite(pth) and pth > 300) or
                      (np.isfinite(phos) and phos > 1.78) or
                      (np.isfinite(ca) and (ca < 2.10 or ca > 2.50)))
        if mbd_signal:
            return _sg(
                "primary", "medium", "2-4 weeks",
                f"CKD-MBD may be impairing erythropoiesis ({tag})",
                f"PTH is {_fmt(pth, 'pg/mL', 0)}, phosphorus is {_fmt(phos, 'mmol/L', 2)}, and calcium is {_fmt(ca, 'mmol/L', 2)}.",
                [
                    "Review phosphate binder timing with meals and dietary phosphate sources",
                    "Assess active vitamin D or analog therapy, calcimimetic suitability, and calcium load",
                    "Interpret PTH using the KDIGO CKD G5D range of approximately 2-9 times the assay upper limit rather than a fixed 150-300 pg/mL target",
                ],
                "Avoid over-suppressing PTH or increasing calcium load without reviewing calcium-phosphorus balance and vascular calcification risk.",
                "Repeat calcium, phosphorus, and PTH in 4-8 weeks after therapy changes; Hb and ERI can be reassessed over 4-8 weeks.",
                feature=feat, evidence="KDIGO CKD-MBD 2017", shap=sv,
            )

    if feat in ("current_pre_sbp_mean", "current_pre_dbp_mean") and np.isfinite(sbp):
        if sbp < 110:
            return _sg(
                "primary", "high", "Immediate / same day" if ctx["idh"] else "Within 1 week",
                f"Low pre-dialysis blood pressure may limit dialysis delivery ({tag})",
                f"Pre-dialysis BP is {_fmt(sbp, 'mmHg', 0)}/{_fmt(dbp, 'mmHg', 0)} and IDH is {_yes_no(ctx['idh'])}.",
                [
                    "Assess dry weight, ultrafiltration rate, interdialytic weight gain, and antihypertensive timing",
                    "Evaluate cardiac function or autonomic dysfunction if instability persists",
                    "Stabilize hemodynamics so dialysis adequacy and anemia management can be delivered safely",
                ],
                "Avoid aggressive ultrafiltration or pre-dialysis antihypertensive dosing when recurrent IDH is present.",
                "Track pre-/intra-dialysis BP, IDH events, achieved treatment time, Kt/V, and Hb over the next month.",
                feature="current_pre_sbp_mean", evidence="KDOQI hemodialysis practice guidance", shap=sv,
            )
        if sbp > 160:
            return _sg(
                "primary", "medium", "2-4 weeks",
                f"Hypertension and volume status require review ({tag})",
                f"Pre-dialysis SBP is {_fmt(sbp, 'mmHg', 0)}; volume overload can coexist with inflammation, underdialysis, and ESA exposure.",
                [
                    "Review interdialytic weight gain, sodium intake, dry weight, and antihypertensive adherence",
                    "Assess whether volume management can be improved without provoking IDH",
                ],
                "Avoid intensifying antihypertensives without considering dry weight and dialysis tolerance.",
                "Monitor BP trend, IDH, ultrafiltration rate, and Hb response over 2-4 weeks.",
                feature="current_pre_sbp_mean", evidence="KDOQI hemodialysis practice guidance", shap=sv,
            )

    if feat in ("current_idh_any", "idh_any_q1", "idh_count_q1") and ctx["idh"]:
        return _sg(
            "primary", "high", "Within 1 week",
            f"Intradialytic hypotension should be corrected to protect dialysis delivery ({tag})",
            f"IDH is present with pre-dialysis BP {_fmt(sbp, 'mmHg', 0)}/{_fmt(dbp, 'mmHg', 0)}, Kt/V {_fmt(ktv, digits=2)}, albumin {_fmt(alb, 'g/L')}.",
            [
                "Re-evaluate dry weight and ultrafiltration rate; target UF rate below 10 mL/kg/h when feasible",
                "Review pre-dialysis antihypertensive timing and interdialytic weight gain",
                "Consider cool dialysate, sodium profiling, or midodrine for recurrent IDH according to local protocol",
                "Assess cardiac function if IDH is persistent or unexplained",
            ],
            "Avoid shortening dialysis repeatedly as the only response to IDH because it worsens delivered adequacy.",
            "Review IDH events each treatment; repeat monthly Kt/V/URR and reassess Hb/ERI after stabilization.",
            feature="current_idh_any", evidence="KDOQI hemodialysis practice guidance", shap=sv,
        )

    if feat == "iron_use_flag" and not ctx["iron_use"]:
        return _sg(
            "primary", "medium", "2-4 weeks",
            f"Iron therapy status should be reconciled ({tag})",
            f"No iron use is recorded and the current iron pattern is {ctx['iron_pattern']}.",
            [
                "Confirm whether iron therapy was intentionally withheld, missed, or contraindicated",
                "Use ferritin, TSAT, CRP, and infection status to decide iron strategy",
                "Reassess ESA response after iron status is corrected",
            ],
            "Avoid empiric iron when active infection is suspected or ferritin is very high without protocol review.",
            "Repeat ferritin, TSAT, and Hb in 4-8 weeks after iron strategy changes.",
            feature="iron_use_flag", evidence="KDIGO anemia guidance", shap=sv,
        )

    label = FEATURE_LABELS.get(feat, feat)
    return _sg(
        "primary", "medium", "2-4 weeks",
        f"Review {label} in the full clinical context ({tag})",
        f"{label} contributes to the model's predicted risk; interpret alongside Hb, ERI, iron status, inflammation, dialysis adequacy, MBD, and hemodynamics.",
        ["Confirm data accuracy", "Assess whether the factor is modifiable", "Integrate with the higher-priority clinical action list"],
        "Avoid acting on model attribution alone without clinical confirmation.",
        "Reassess after correcting higher-priority reversible drivers.",
        feature=feat, evidence="Model attribution plus clinical assessment", shap=sv,
    )


def _urgent_rules(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    suggestions = []
    crp = ctx["crp"]
    ferritin = ctx["ferritin"]
    tsat = ctx["tsat"]
    albumin = ctx["albumin"]
    sbp = ctx["sbp"]
    dbp = ctx["dbp"]
    potassium = ctx["potassium"]
    sodium = ctx["sodium"]

    if np.isfinite(crp) and crp > 50:
        iron_text = f", TSAT {_fmt(tsat, '%')} and ferritin {_fmt(ferritin, 'ng/mL', 0)}" if np.isfinite(tsat) or np.isfinite(ferritin) else ""
        suggestions.append(_sg(
            "urgent", "emergency", "Immediate / same day",
            "Screen for active infection or severe inflammation before anemia escalation",
            f"CRP is {_fmt(crp, 'mg/L')}{iron_text}; this pattern can cause inflammation-mediated ESA resistance and functional iron restriction.",
            [
                "Assess vascular access for tenderness, erythema, drainage, or dysfunction",
                "Ask about fever, chills, respiratory, urinary, skin, wound, gastrointestinal, and dental symptoms",
                "Obtain cultures or imaging when clinical findings support infection workup",
                "Treat the inflammatory source before interpreting ESA failure",
            ],
            "Avoid simply increasing ESA during uncontrolled inflammation or suspected active infection.",
            "Recheck CRP, Hb, ESA dose, ERI, ferritin, and TSAT within 2-4 weeks, or sooner if clinically unstable.",
            feature="crp", evidence="KDIGO anemia guidance; clinical infection workflow",
        ))

    if ctx["idh"] and np.isfinite(sbp) and sbp < 110:
        suggestions.append(_sg(
            "urgent", "emergency", "Immediate / same day",
            "Stabilize low pre-dialysis BP with intradialytic hypotension",
            f"IDH is present with pre-dialysis BP {_fmt(sbp, 'mmHg', 0)}/{_fmt(dbp, 'mmHg', 0)}; this can cause end-organ hypoperfusion, shortened dialysis, and low delivered clearance.",
            [
                "Review dry weight, interdialytic weight gain, ultrafiltration rate, and recent symptoms today",
                "Adjust antihypertensive timing and consider holding pre-dialysis doses when appropriate",
                "Reduce UF rate when feasible and consider cool dialysate, sodium profiling, or midodrine for recurrent IDH",
                "Assess cardiac function if hypotension is recurrent or unexplained",
            ],
            "Avoid repeated treatment shortening as the only response to hypotension.",
            "Track IDH every treatment and reassess delivered Kt/V or URR after hemodynamic stabilization.",
            feature="current_idh_any", evidence="KDOQI hemodialysis practice guidance",
        ))

    if np.isfinite(albumin) and albumin < 25:
        suggestions.append(_sg(
            "urgent", "emergency", "Immediate / same day",
            "Treat severe hypoalbuminemia as a nutrition-inflammation emergency",
            f"Albumin is {_fmt(albumin, 'g/L')}, which indicates severe protein-energy wasting or inflammation and carries immediate clinical risk beyond anemia.",
            [
                "Request dietitian assessment and review protein-energy intake",
                "Evaluate edema, liver disease, gastrointestinal loss, inflammation, infection, and recent hospitalization",
                "Coordinate nutrition support with inflammation workup and dialysis adequacy review",
            ],
            "Avoid focusing anemia management on ESA dose while severe hypoalbuminemia remains unexplained.",
            "Recheck albumin, CRP, weight trend, dietary intake, Hb, and ERI within 2-4 weeks.",
            feature="albumin", evidence="KDOQI Nutrition 2020",
        ))

    if np.isfinite(potassium) and potassium >= 6.0:
        suggestions.append(_sg(
            "urgent", "emergency", "Immediate / same day",
            "Manage hyperkalemia before routine anemia optimization",
            f"Potassium is {_fmt(potassium, 'mmol/L')}, which should be handled through dialysis-unit safety protocols before non-urgent ESA decisions.",
            [
                "Confirm sample validity and assess ECG or symptoms according to local emergency workflow",
                "Review recent dialysis adequacy, missed treatment, diet, medications, and access function",
                "Use the unit's hyperkalemia management protocol before anemia medication changes",
            ],
            "Avoid delaying hyperkalemia management to address ESA hyporesponsiveness first.",
            "Recheck potassium according to local protocol and review dialysis adequacy at the next assessment.",
            feature="potassium", evidence="Clinical safety workflow",
        ))

    if np.isfinite(sodium) and (sodium < 125 or sodium > 150):
        suggestions.append(_sg(
            "urgent", "emergency", "Immediate / same day",
            "Address marked sodium abnormality before anemia optimization",
            f"Sodium is {_fmt(sodium, 'mmol/L', 0)}, which requires safety review before elective anemia treatment changes.",
            [
                "Confirm the result and assess volume status, neurological symptoms, glucose, and recent dialysate sodium exposure",
                "Manage according to local electrolyte and dialysis safety protocols",
            ],
            "Avoid treating the anemia plan as the immediate priority until electrolyte safety is addressed.",
            "Recheck sodium according to local safety protocol and reassess dialysis prescription if recurrent.",
            feature="sodium", evidence="Clinical safety workflow",
        ))

    return suggestions


def _reversible_cause_rules(ctx: Dict[str, Any], shap_result: Dict) -> List[Dict[str, Any]]:
    suggestions = []
    hb = ctx["hb"]
    eri = ctx["eri"]
    crp = ctx["crp"]
    albumin = ctx["albumin"]
    ktv = ctx["ktv"]
    urr = ctx["urr"]
    pth = ctx["pth"]
    calcium = ctx["calcium"]
    phosphorus = ctx["phosphorus"]
    ferritin = ctx["ferritin"]
    tsat = ctx["tsat"]

    if ctx["ferritin_missing"] or ctx["tsat_missing"] or not np.isfinite(ferritin) or not np.isfinite(tsat):
        suggestions.append(_sg(
            "primary", "high", "Within 1 week",
            "Complete iron-status testing before classifying ESA low response",
            "Ferritin and TSAT are both required to distinguish absolute iron deficiency from functional iron restriction.",
            [
                "Order ferritin and TSAT together",
                "Review recent iron administration, transfusion, infection, inflammation, and blood loss",
                "Use the updated iron pattern to decide iron, ESA, and HIF-PHI strategy",
            ],
            "Avoid increasing ESA or giving empiric iron without current iron indices unless clinically justified.",
            "Repeat iron indices 4-8 weeks after any iron intervention.",
            feature="iron_status", evidence="KDIGO anemia guidance",
        ))
    elif tsat < 20 and ferritin < 200:
        suggestions.append(_sg(
            "primary", "high", "Within 1 week",
            "Prioritize absolute iron deficiency correction",
            f"TSAT is {_fmt(tsat, '%')} and ferritin is {_fmt(ferritin, 'ng/mL', 0)}, consistent with depleted iron stores and poor substrate availability for erythropoiesis.",
            [
                "Assess recent blood loss, access bleeding, gastrointestinal bleeding risk, and iron adherence",
                "Optimize iron repletion according to local hemodialysis protocol",
                "Review Hb response before additional ESA escalation",
            ],
            "Avoid labeling the patient ESA-resistant before iron stores are corrected.",
            "Repeat Hb, ferritin, TSAT, and ESA dose in 4-8 weeks.",
            feature="tsat_mean", evidence="KDIGO anemia guidance",
        ))
    elif tsat < 20 and ferritin >= 200:
        timeframe = "Within 1 week" if (np.isfinite(crp) and crp > 5) or ferritin > 800 else "2-4 weeks"
        suggestions.append(_sg(
            "primary", "high" if timeframe == "Within 1 week" else "medium", timeframe,
            "Treat functional iron restriction as inflammation-aware ESA resistance",
            f"TSAT is {_fmt(tsat, '%')} with ferritin {_fmt(ferritin, 'ng/mL', 0)} and CRP {_fmt(crp, 'mg/L')}; this favors inflammation-mediated iron sequestration when inflammation is present.",
            [
                "Screen for infection or chronic inflammation before changing iron or ESA",
                "Decide whether iron should be continued, paused, or adjusted using ferritin, TSAT, infection status, and local protocol",
                "Correct inflammation first when CRP is elevated",
            ],
            "Avoid additional iron or ESA escalation as a default when ferritin is high or infection is suspected.",
            "Repeat CRP, ferritin, TSAT, Hb, and ERI in 2-4 weeks if active inflammation; otherwise 4-8 weeks.",
            feature="tsat_mean", evidence="KDIGO anemia guidance; inflammation-mediated iron restriction literature",
        ))
    elif ferritin < 200:
        suggestions.append(_sg(
            "primary", "medium", "2-4 weeks",
            "Review low iron stores even if TSAT is not severely reduced",
            f"Ferritin is {_fmt(ferritin, 'ng/mL', 0)} with TSAT {_fmt(tsat, '%')}; depleted stores can limit sustained Hb response.",
            [
                "Confirm iron dosing history and recent interruptions",
                "Assess chronic blood loss and access bleeding",
                "Optimize iron therapy before changing long-term ESA strategy",
            ],
            "Avoid assuming Hb will respond to ESA alone when iron stores are depleted.",
            "Repeat Hb, ferritin, and TSAT in 4-8 weeks.",
            feature="ferritin_mean", evidence="KDIGO anemia guidance",
        ))

    if np.isfinite(crp) and 5 < crp <= 50:
        suggestions.append(_sg(
            "primary", "high" if crp >= 20 else "medium", "Within 1 week" if crp >= 20 else "2-4 weeks",
            "Identify and treat inflammation driving ESA hyporesponsiveness",
            f"CRP is {_fmt(crp, 'mg/L')}, albumin is {_fmt(albumin, 'g/L')}, and iron pattern is {ctx['iron_pattern']}.",
            [
                "Review vascular access, recent infection, wounds, respiratory and urinary symptoms, dental disease, heart failure, autoimmune disease, and malignancy clues",
                "Use targeted cultures, imaging, or referral when the clinical review suggests a source",
                "Reassess anemia after inflammation improves",
            ],
            "Avoid escalating ESA as the sole response to inflammatory anemia.",
            "Repeat CRP, Hb, ferritin, TSAT, and ERI in 2-4 weeks.",
            feature="crp", evidence="KDIGO anemia guidance; clinical best practice",
        ))

    if np.isfinite(albumin) and 25 <= albumin < 35:
        suggestions.append(_sg(
            "primary", "high" if albumin < 30 else "medium", "Within 1 week" if albumin < 30 else "2-4 weeks",
            "Correct nutrition-inflammation burden that may blunt ESA response",
            f"Albumin is {_fmt(albumin, 'g/L')}; low albumin can reflect protein-energy wasting, inflammation, or fluid overload and is linked to poor ESA response.",
            [
                "Assess dietary intake, appetite, gastrointestinal symptoms, edema, inflammation, and dialysis adequacy",
                "Request dietitian review and set individualized protein-energy targets",
                "Coordinate nutrition intervention with infection and dialysis adequacy review",
            ],
            "Avoid focusing only on anemia medication while malnutrition-inflammation drivers remain active.",
            "Recheck albumin, CRP, Hb, and ERI in 2-4 weeks.",
            feature="albumin", evidence="KDOQI Nutrition 2020",
        ))

    if (np.isfinite(ktv) and ktv < 1.2) or (np.isfinite(urr) and urr < 65):
        suggestions.append(_sg(
            "primary", "high", "Within 1 week",
            "Optimize dialysis adequacy as a reversible ESA low-response driver",
            f"Kt/V is {_fmt(ktv, digits=2)} and URR is {_fmt(urr, '%')}; underdialysis can worsen inflammation and erythropoietic suppression.",
            [
                "Check delivered treatment time, missed or shortened sessions, blood flow, dialyzer clearance, access function, and recirculation",
                "Address IDH or access problems that reduce delivered dose",
                "Consider prescription adjustment or longer treatment time if targets remain unmet",
            ],
            "Avoid treating low Hb only with ESA when delivered dialysis is below target.",
            "Repeat Kt/V, URR, IDH events, Hb, and ERI after the next monthly adequacy cycle.",
            feature="ktv", evidence="KDOQI hemodialysis adequacy guidance",
        ))

    mbd_signal = ((np.isfinite(pth) and pth > 300) or
                  (np.isfinite(phosphorus) and phosphorus > 1.78) or
                  (np.isfinite(calcium) and (calcium < 2.10 or calcium > 2.50)))
    if mbd_signal:
        suggestions.append(_sg(
            "primary", "medium", "2-4 weeks",
            "Address CKD-MBD contributors to impaired erythropoiesis",
            f"PTH is {_fmt(pth, 'pg/mL', 0)}, phosphorus is {_fmt(phosphorus, 'mmol/L', 2)}, and calcium is {_fmt(calcium, 'mmol/L', 2)}.",
            [
                "Review dietary phosphate sources and phosphate binder timing with meals",
                "Assess active vitamin D or analog therapy, calcimimetic suitability, and calcium exposure",
                "Use KDIGO CKD G5D guidance: maintain PTH approximately 2-9 times the assay upper limit rather than a fixed 150-300 pg/mL target",
                "Consider nephrology team escalation when PTH or phosphorus remains markedly elevated",
            ],
            "Avoid over-suppression of PTH or excess calcium loading without reviewing vascular calcification and calcium-phosphorus balance.",
            "Repeat calcium and phosphorus in 4-8 weeks, and PTH in 4-8 or 8-12 weeks depending on therapy intensity and local protocol.",
            feature="pth", evidence="KDIGO CKD-MBD 2017",
        ))

    composite = (
        np.isfinite(crp) and crp > 5 and
        np.isfinite(albumin) and albumin < 35 and
        ((np.isfinite(ktv) and ktv < 1.2) or (np.isfinite(urr) and urr < 65)) and
        ((np.isfinite(hb) and hb < 100) or (np.isfinite(eri) and eri > 12))
    )
    if composite:
        suggestions.append(_sg(
            "primary", "high", "Within 1 week",
            "Use an integrated inflammation-malnutrition-underdialysis pathway",
            f"The patient has CRP {_fmt(crp, 'mg/L')}, albumin {_fmt(albumin, 'g/L')}, Kt/V {_fmt(ktv, digits=2)}, URR {_fmt(urr, '%')}, Hb {_fmt(hb, 'g/L')}, and ERI {_fmt(eri)}.",
            [
                "First screen and treat infection or inflammatory sources",
                "Second initiate nutrition assessment and support",
                "Third correct dialysis delivery barriers such as shortened treatments, access dysfunction, or IDH",
                "Only then reassess whether ESA or HIF-PHI strategy should change",
            ],
            "Avoid addressing these abnormalities as isolated checklist items; the combined pattern often explains ESA hyporesponsiveness.",
            "Repeat CRP, albumin, Kt/V or URR, Hb, ESA dose, and ERI within 2-4 weeks, then adjust the pathway.",
            feature="composite", evidence="Clinical integration of KDIGO/KDOQI guidance",
        ))

    return suggestions


def _esa_strategy_rules(ctx: Dict[str, Any], risk_level: str, risk_score: float) -> List[Dict[str, Any]]:
    suggestions = []
    hb = ctx["hb"]
    eri = ctx["eri"]
    esa_dose = ctx["esa_dose"]
    reversible_active = bool(
        (np.isfinite(ctx["crp"]) and ctx["crp"] > 5) or
        (np.isfinite(ctx["albumin"]) and ctx["albumin"] < 35) or
        (np.isfinite(ctx["ktv"]) and ctx["ktv"] < 1.2) or
        (np.isfinite(ctx["urr"]) and ctx["urr"] < 65) or
        (np.isfinite(ctx["tsat"]) and ctx["tsat"] < 20) or
        (np.isfinite(ctx["ferritin"]) and ctx["ferritin"] < 200) or
        (np.isfinite(ctx["pth"]) and ctx["pth"] > 300) or
        (np.isfinite(ctx["phosphorus"]) and ctx["phosphorus"] > 1.78) or
        ctx["idh"]
    )

    if (np.isfinite(eri) and eri > 12) or (np.isfinite(esa_dose) and esa_dose > 12000):
        suggestions.append(_sg(
            "supportive", "medium", "2-4 weeks",
            "ESA strategy: complete reversible-cause checklist before dose escalation",
            f"ERI is {_fmt(eri)} and ESA dose is {_fmt(esa_dose, 'IU/week', 0)}; high exposure with low response should trigger a workup-first strategy.",
            [
                "Document the active reversible drivers and whether each has a plan",
                "Use local anemia protocol for any conservative ESA adjustment after iron, inflammation, dialysis adequacy, MBD, and bleeding review",
                "Discuss risk-benefit when ESA exposure is already high",
            ],
            "Avoid repeated ESA increases when Hb response is poor and reversible drivers remain active.",
            "Reassess Hb, ESA dose, ERI, blood pressure, and thrombotic or cardiovascular events in 2-4 weeks.",
            feature="esa_dose", evidence="KDIGO anemia guidance; ESA safety communications",
        ))

    if np.isfinite(hb) and hb < 100 and not ((np.isfinite(eri) and eri > 12) or (np.isfinite(esa_dose) and esa_dose > 12000)):
        suggestions.append(_sg(
            "supportive", "medium", "2-4 weeks",
            "ESA strategy: consider modest protocol-based adjustment only after checks",
            f"Hb is {_fmt(hb, 'g/L')}; ESA dose and ERI are not clearly excessive, but reversible contributors are {'active' if reversible_active else 'not prominent'}.",
            [
                "Confirm iron status, inflammation, occult bleeding, hemolysis, dialysis adequacy, and CKD-MBD status",
                "If those checks are addressed, consider a modest ESA adjustment using the local anemia protocol",
            ],
            "Avoid adjusting ESA based on Hb alone without reviewing correctable contributors.",
            "Repeat Hb and ESA dose response in 2-4 weeks after medication changes.",
            feature="hb", evidence="KDIGO anemia guidance",
        ))

    if np.isfinite(hb) and 100 <= hb <= 115:
        suggestions.append(_sg(
            "supportive", "low", "4-8 week reassessment",
            "ESA strategy: Hb is in the usual maintenance range",
            f"Hb is {_fmt(hb, 'g/L')}; model risk should be used to address risk factors rather than to push Hb higher.",
            [
                "Maintain anemia therapy unless local protocol or symptoms indicate otherwise",
                "Focus on modifiable risk factors driving future ESA hyporesponsiveness",
            ],
            "Avoid increasing ESA solely because model risk is intermediate or high when Hb is already acceptable.",
            "Repeat Hb, ESA dose, and ERI at routine interval or within 4-8 weeks after risk-factor interventions.",
            feature="hb", evidence="KDIGO anemia guidance",
        ))

    if np.isfinite(hb) and hb > 115:
        suggestions.append(_sg(
            "supportive", "medium", "2-4 weeks",
            "ESA strategy: avoid hemoglobin overcorrection",
            f"Hb is {_fmt(hb, 'g/L')}; anemia treatment intensity should be reviewed to avoid excessive Hb correction.",
            [
                "Review ESA dose trajectory and cardiovascular or thrombotic risk",
                "Consider dose reduction or holding strategy according to local protocol if Hb continues to rise",
            ],
            "Avoid intensifying ESA or HIF-PHI when Hb is above the usual maintenance range.",
            "Repeat Hb within 2-4 weeks if therapy is adjusted.",
            feature="hb", evidence="KDIGO anemia guidance; ESA safety communications",
        ))

    if not ctx["hif_use"] and risk_level in ("High", "Very High"):
        suggestions.append(_sg(
            "supportive", "low", "4-8 week reassessment",
            "HIF-PHI is a conditional discussion option, not a default substitution",
            f"Predicted ESA hyporesponsiveness risk is {risk_score:.1%} ({risk_level}); HIF-PHI may be considered only after reversible factors are corrected or actively managed.",
            [
                "Discuss HIF-PHI with the nephrology team only if ESA hyporesponsiveness persists",
                "Review contraindications, cardiovascular and thrombotic risk, malignancy history, drug interactions, and local formulary criteria",
                "Continue correcting iron restriction, inflammation, underdialysis, MBD, and nutrition drivers in parallel",
            ],
            "Avoid presenting HIF-PHI as automatic replacement therapy or using it before reversible drivers are addressed.",
            "If started, monitor Hb trajectory, iron indices, blood pressure, and adverse effects according to local protocol.",
            feature="hif_use_flag", evidence="Expert consensus; HIF-PHI clinical literature",
        ))

    return suggestions


def _phenotype_rules(ctx: Dict[str, Any], phenotype: str) -> List[Dict[str, Any]]:
    idx = None
    for k, name in SUBTYPE_NAMES.items():
        if name == phenotype:
            idx = k
            break
    if idx is None:
        return []

    if idx == 0:
        title = "Phenotype focus: Hemodynamic instability"
        rationale = f"Assigned phenotype is Hemodynamic instability; IDH is {_yes_no(ctx['idh'])}, pre-dialysis BP is {_fmt(ctx['sbp'], 'mmHg', 0)}/{_fmt(ctx['dbp'], 'mmHg', 0)}, and delivered Kt/V is {_fmt(ctx['ktv'], digits=2)}."
        actions = [
            "Prioritize dry-weight, UF-rate, antihypertensive timing, and cardiac review",
            "Use cool dialysate, sodium profiling, or midodrine only when appropriate under local protocol",
            "Recheck delivered dialysis adequacy after hemodynamic stabilization",
        ]
        avoid = "Avoid sacrificing dialysis time repeatedly without fixing the reason for instability."
        monitoring = "Track IDH every session; repeat Kt/V/URR monthly and Hb/ERI after stability improves."
    elif idx == 1:
        title = "Phenotype focus: MBD-dominant"
        rationale = f"Assigned phenotype is MBD-dominant; PTH is {_fmt(ctx['pth'], 'pg/mL', 0)}, phosphorus is {_fmt(ctx['phosphorus'], 'mmol/L', 2)}, and calcium is {_fmt(ctx['calcium'], 'mmol/L', 2)}."
        actions = [
            "Optimize phosphate binder use with meals and dietary phosphate intake",
            "Review active vitamin D or analog therapy, calcimimetic eligibility, and calcium loading",
            "Use the KDIGO CKD G5D PTH range of approximately 2-9 times the assay upper limit",
        ]
        avoid = "Avoid using a fixed 150-300 pg/mL PTH target as a universal CKD G5D treatment goal."
        monitoring = "Repeat calcium/phosphorus in 4-8 weeks and PTH in 4-8 or 8-12 weeks depending on intervention."
    else:
        title = "Phenotype focus: inflammation-malnutrition-underdialysis"
        rationale = f"Assigned phenotype is inflammation-malnutrition-underdialysis; CRP is {_fmt(ctx['crp'], 'mg/L')}, albumin is {_fmt(ctx['albumin'], 'g/L')}, Kt/V is {_fmt(ctx['ktv'], digits=2)}, URR is {_fmt(ctx['urr'], '%')}, and iron pattern is {ctx['iron_pattern']}."
        actions = [
            "Screen for infection or chronic inflammation first",
            "Start nutrition review and support when albumin is low",
            "Correct delivered dialysis barriers including treatment time, blood flow, access function, and IDH",
            "Reassess ESA or HIF-PHI strategy after these drivers are managed",
        ]
        avoid = "Avoid listing CRP, albumin, Kt/V, and iron abnormalities separately without an integrated sequence."
        monitoring = "Repeat CRP, albumin, ferritin, TSAT, Kt/V/URR, Hb, ESA dose, and ERI in 2-4 weeks if high risk."

    return [_sg(
        "phenotype", "medium", "2-4 weeks",
        title, rationale, actions, avoid, monitoring,
        feature="phenotype", evidence="Phenotype clustering analysis integrated with clinical rules",
    )]


def _followup_plan(ctx: Dict[str, Any], risk_level: str) -> List[Dict[str, Any]]:
    if risk_level in ("High", "Very High"):
        timeframe = "2-4 weeks"
        severity = "medium"
        actions = [
            "Create a problem list ordered as emergency issues, reversible causes, anemia medication strategy, then routine monitoring",
            "Repeat Hb, ESA dose, ERI, CRP, ferritin, TSAT, albumin, and any abnormal safety labs",
            "If dialysis prescription changed, review the next Kt/V or URR result and IDH trend",
        ]
        rationale = f"Risk level is {risk_level}; short-interval reassessment is needed to confirm that the highest-priority drivers are improving."
    elif risk_level == "Intermediate":
        timeframe = "4-8 week reassessment"
        severity = "low"
        actions = [
            "Recheck Hb, ESA dose, ERI, iron indices, CRP, and the active phenotype-specific drivers",
            "Escalate to a high-risk review pathway if Hb falls, ERI rises, CRP increases, or dialysis adequacy worsens",
        ]
        rationale = "Risk level is Intermediate; monitoring should verify that modifiable risks do not progress."
    else:
        timeframe = "4-8 week reassessment"
        severity = "low"
        actions = [
            "Continue routine Hb, iron-status, dialysis adequacy, and CKD-MBD monitoring",
            "Repeat model assessment when clinical parameters change or at the next scheduled review",
        ]
        rationale = "Risk level is Low; routine surveillance is appropriate unless new abnormalities appear."

    return [_sg(
        "supportive", severity, timeframe,
        "Follow-up and reassessment plan",
        rationale, actions,
        "Avoid changing anemia therapy without a documented response assessment interval.",
        "Use the reassessment interval above; iron interventions usually need 4-8 weeks, MBD interventions 4-8 or 8-12 weeks, and dialysis changes the next adequacy cycle.",
        feature=None, evidence="Clinical best practice",
    )]


def _rank_suggestions(suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(
        suggestions,
        key=lambda s: (
            SEVERITY_RANK.get(s.get("severity", "low"), 9),
            TIMEFRAME_RANK.get(s.get("timeframe", "4-8 week reassessment"), 9),
            TIER_RANK.get(s.get("tier", "supportive"), 9),
            s.get("priority", 999),
            -abs(s.get("shap") or 0),
            s.get("title", ""),
        )
    )
    for i, sg in enumerate(ranked, 1):
        sg["priority"] = i
    return ranked


def _dedupe_suggestions(suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for sg in suggestions:
        key = (sg.get("tier"), sg.get("feature"), sg.get("title"))
        broad_key = (sg.get("feature"), sg.get("severity"), sg.get("timeframe"))
        if key in seen or broad_key in seen:
            continue
        seen.add(key)
        seen.add(broad_key)
        deduped.append(sg)
    return deduped


def generate_suggestions(values: Dict, shap_result: Dict, phenotype: str,
                         risk_level: str, risk_score: float) -> List[Dict]:
    """Generate patient-level clinical action recommendations."""
    ctx = _patient_context(values)
    suggestions = []
    priority_counter = 0

    def add(items):
        nonlocal priority_counter
        for item in items:
            priority_counter += 1
            item["priority"] = priority_counter
            suggestions.append(item)

    add(_urgent_rules(ctx))
    add(_reversible_cause_rules(ctx, shap_result))

    seen_domains = {s.get("feature") for s in suggestions if s.get("feature")}
    seen_shap_domains = set()
    for d in shap_result.get("drivers", [])[:10]:
        feat = d["feature"]
        dom = _domain(feat)
        if dom in seen_shap_domains:
            continue
        if feat in seen_domains and dom in ("iron_status", "dialysis", "mbd", "crp", "albumin"):
            seen_shap_domains.add(dom)
            continue
        sg = _driver_suggestion(feat, d["shap"], values)
        if sg:
            priority_counter += 1
            sg["priority"] = priority_counter
            suggestions.append(sg)
            seen_shap_domains.add(dom)

    add(_esa_strategy_rules(ctx, risk_level, risk_score))
    add(_phenotype_rules(ctx, phenotype))
    add(_followup_plan(ctx, risk_level))

    suggestions = _dedupe_suggestions(suggestions)
    return _rank_suggestions(suggestions)


# ---------------------------------------------------------------------------
# Main prediction entry point
# ---------------------------------------------------------------------------

def predict_case(values: Dict[str, Any]) -> Dict[str, Any]:
    """Full CDSS prediction pipeline: risk + phenotype + SHAP + recommendations."""
    assets = get_assets()

    df = build_input_dataframe(values)
    X_processed = assets.transform(df)
    prob = float(assets.xgb_model.predict_proba(X_processed)[0, 1])
    risk_level = assign_risk_level(prob)

    row = df.iloc[0].to_dict()
    pheno = assign_phenotype(row, assets.cluster_meta)
    shap_result = compute_shap(assets, df)
    suggestions = generate_suggestions(
        row, shap_result, pheno["assigned"], risk_level, prob
    )

    return {
        "risk_score": prob,
        "risk_level": risk_level,
        "risk_color": RISK_COLORS.get(risk_level, "#95a5a6"),
        "risk_description": RISK_DESCRIPTIONS.get(risk_level, ""),
        "phenotype": pheno["assigned"],
        "phenotype_short": pheno["assigned_short"],
        "phenotype_index": pheno["assigned_index"],
        "phenotype_description": pheno["description"],
        "phenotype_checklist": pheno["checklist"],
        "phenotype_distances": pheno["distances"],
        "shap": shap_result,
        "suggestions": suggestions,
        "input_values": values,
    }
