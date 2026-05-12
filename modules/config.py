"""
ESA CDSS Configuration
======================
Central configuration file for the Clinical Decision Support System.
Contains all constants, thresholds, feature definitions, and phenotype names.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
FIGURES_DIR = BASE_DIR / "figures"
VALIDATION_DIR = BASE_DIR / "validation"
OUTPUT_DIR = BASE_DIR / "output"

MODEL_PATH = ASSETS_DIR / "best_model_XGBoost.joblib"
CLUSTER_META_PATH = ASSETS_DIR / "cluster_metadata.json"

# ---------------------------------------------------------------------------
# Phenotype names (05-09 revised clustering with iron-status variables)
# ---------------------------------------------------------------------------

SUBTYPE_NAMES = {
    0: "Hemodynamic instability",
    1: "MBD-dominant",
    2: "inflammation-malnutrition-underdialysis",
}

SUBTYPE_SHORT_NAMES = {
    0: "Hemodynamic instability",
    1: "MBD-dominant",
    2: "inflammation-malnutrition-underdialysis",
}

SUBTYPE_DESCRIPTIONS = {
    0: (
        "Characterized by intradialytic hypotension, lower pre-dialysis blood pressure, "
        "and hemodynamic instability during dialysis sessions, with a mixed anemia and "
        "metabolic profile. This phenotype prioritizes volume, blood pressure, and "
        "dialysis-tolerance optimization."
    ),
    1: (
        "Characterized by CKD-MBD burden with relatively preserved nutrition and dialysis "
        "adequacy. Mineral metabolism abnormalities and secondary hyperparathyroidism may "
        "contribute to impaired erythropoiesis and ESA hyporesponsiveness."
    ),
    2: (
        "Characterized by lower hemoglobin, higher ESA requirement and ERI, inflammatory "
        "activity, weaker nutritional indices, suboptimal dialysis adequacy, and iron-status "
        "patterns consistent with absolute or functional iron restriction."
    ),
}

SUBTYPE_CHECKLISTS = {
    0: [
        "Re-evaluate target dry weight using clinical assessment or bioimpedance if available",
        "Reduce ultrafiltration rate to < 10 mL/kg/h when feasible",
        "Review interdialytic weight gain and reinforce fluid restriction counseling",
        "Adjust antihypertensive timing, including withholding pre-dialysis doses when appropriate",
        "Consider cool dialysate (35-35.5 C) or sodium profiling for recurrent IDH",
        "Assess cardiac function with echocardiography when instability persists",
        "Recheck delivered dialysis adequacy after hemodynamic stabilization",
    ],
    1: [
        "Optimize PTH toward the KDIGO CKD G5D range of approximately 2-9 times the assay upper limit",
        "Improve phosphate control with dietary review and binder optimization",
        "Review vitamin D status and active vitamin D or analog therapy",
        "Evaluate calcium-phosphorus balance and vascular calcification risk",
        "Consider calcimimetic therapy for refractory secondary hyperparathyroidism",
        "Reassess hemoglobin and ESA response after CKD-MBD optimization",
    ],
    2: [
        "Check ferritin and TSAT to distinguish absolute from functional iron deficiency",
        "Screen for infection or inflammation: vascular access, blood cultures, urine, respiratory, and dental sources",
        "Assess nutritional status with dietary intake, nPCR, and body composition when available",
        "Verify delivered dialysis dose, treatment time, blood flow, dialyzer performance, and access function",
        "Address inflammation and iron restriction before ESA dose escalation",
        "Consider HIF-PHI only after reversible causes are corrected or actively managed and contraindications are excluded",
    ],
}

# ---------------------------------------------------------------------------
# Risk stratification thresholds
# ---------------------------------------------------------------------------

RISK_THRESHOLDS = {
    "Low": (0, 0.20),
    "Intermediate": (0.20, 0.50),
    "High": (0.50, 0.80),
    "Very High": (0.80, 1.01),
}

RISK_COLORS = {
    "Low": "#27ae60",
    "Intermediate": "#f39c12",
    "High": "#e74c3c",
    "Very High": "#8e44ad",
}

RISK_DESCRIPTIONS = {
    "Low": (
        "Low predicted risk of ESA hyporesponsiveness in the next quarter. "
        "Continue current management and routine monitoring."
    ),
    "Intermediate": (
        "Moderate predicted risk. Review modifiable risk factors "
        "before routine ESA adjustment."
    ),
    "High": (
        "High predicted risk. Systematically evaluate reversible "
        "drivers before ESA dose escalation."
    ),
    "Very High": (
        "Very high predicted risk. Urgent comprehensive workup "
        "required. Address all reversible contributors."
    ),
}

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

INPUT_FEATURES = {
    "demographics": {
        "age": {"label": "Age", "unit": "years", "min": 18, "max": 100, "default": 60,
                "help": "Patient age in years"},
        "dialysis_age": {"label": "Dialysis Vintage", "unit": "months", "min": 0.0,
                         "max": 300.0, "default": 24.0, "help": "Time on dialysis"},
    },
    "anemia_esa": {
        "hb": {"label": "Hemoglobin", "unit": "g/L", "min": 30.0, "max": 180.0,
               "default": 100.0, "ref": "100-115 g/L (KDIGO target)"},
        "esa_dose": {"label": "ESA Weekly Dose", "unit": "IU", "min": 0.0,
                     "max": 50000.0, "default": 10000.0, "step": 500.0},
        "esa_route": {"label": "ESA Route", "options": ["Subcutaneous", "Intravenous"],
                      "default": "Subcutaneous"},
        "dry_weight": {"label": "Dry Weight", "unit": "kg", "min": 20.0, "max": 150.0,
                       "default": 60.0},
    },
    "inflammation_nutrition": {
        "crp": {"label": "CRP", "unit": "mg/L", "min": 0.0, "max": 300.0, "default": 5.0,
                "step": 0.1, "ref": "Normal <5; >50 suggests active infection"},
        "albumin": {"label": "Albumin", "unit": "g/L", "min": 10.0, "max": 60.0,
                    "default": 35.0, "ref": "Target >=35; <25 = severe"},
    },
    "iron_status": {
        "ferritin_mean": {"label": "Ferritin", "unit": "ng/mL", "min": 0.0, "max": 5000.0,
                          "default": 200.0, "step": 10.0,
                          "ref": "Common HD target: >=200 ng/mL; interpret with inflammation"},
        "tsat_mean": {"label": "Transferrin Saturation", "unit": "%", "min": 0.0, "max": 100.0,
                      "default": 25.0, "step": 0.5,
                      "ref": "Common HD target: >=20%; low TSAT suggests iron-restricted erythropoiesis"},
    },
    "dialysis_adequacy": {
        "ktv": {"label": "Kt/V", "unit": "", "min": 0.0, "max": 3.0, "default": 1.2,
                "step": 0.01, "ref": "Target >= 1.2"},
        "urr": {"label": "URR", "unit": "%", "min": 0.0, "max": 100.0, "default": 65.0,
                "ref": "Target >= 65%"},
    },
    "ckd_mbd": {
        "pth": {"label": "PTH", "unit": "pg/mL", "min": 0.0, "max": 3000.0,
                "default": 300.0, "ref": "KDIGO CKD G5D: approximately 2-9x the assay upper limit"},
        "calcium": {"label": "Calcium", "unit": "mmol/L", "min": 1.0, "max": 3.5,
                    "default": 2.2, "step": 0.01, "ref": "Target: 2.10-2.50"},
        "phosphorus": {"label": "Phosphorus", "unit": "mmol/L", "min": 0.0, "max": 4.0,
                       "default": 1.8, "step": 0.01, "ref": "Target: 1.13-1.78"},
    },
    "electrolytes": {
        "potassium": {"label": "Potassium", "unit": "mmol/L", "min": 0.0, "max": 8.0,
                      "default": 4.8, "step": 0.1},
        "sodium": {"label": "Sodium", "unit": "mmol/L", "min": 100.0, "max": 160.0,
                   "default": 138.0},
        "creatinine": {"label": "Creatinine", "unit": "umol/L", "min": 0.0,
                       "max": 2000.0, "default": 800.0},
    },
    "hemodynamics": {
        "sbp": {"label": "Pre-dialysis SBP", "unit": "mmHg", "min": 50.0, "max": 250.0,
                "default": 145.0},
        "dbp": {"label": "Pre-dialysis DBP", "unit": "mmHg", "min": 30.0, "max": 150.0,
                "default": 80.0},
        "idh_any": {"label": "Intradialytic Hypotension (IDH)", "type": "checkbox",
                    "default": False, "ref": "SBP drop >= 20 mmHg or MAP < 70"},
    },
    "medications": {
        "iron_use": {"label": "Iron Supplement", "type": "checkbox", "default": True},
        "hif_use": {"label": "HIF-PHI", "type": "checkbox", "default": False},
    },
}

# SHAP display labels
FEATURE_LABELS = {
    "age": "Age",
    "dialysis_age": "Dialysis Vintage",
    "hb": "Hemoglobin",
    "esa_dose": "ESA Weekly Dose",
    "eq_esa_dose": "Equivalent ESA Dose",
    "eri": "ERI",
    "crp": "CRP",
    "log_crp_w99": "log(CRP)",
    "albumin": "Albumin",
    "ktv": "Kt/V",
    "urr": "URR",
    "pth": "PTH",
    "calcium": "Calcium",
    "phosphorus": "Phosphorus",
    "potassium": "Potassium",
    "sodium": "Sodium",
    "creatinine": "Creatinine",
    "dry_weight": "Dry Weight",
    "iron_use_flag": "Iron Supplement",
    "ferritin": "Ferritin",
    "tsat": "TSAT",
    "ferritin_mean": "Ferritin",
    "ferritin_latest": "Latest Ferritin",
    "tsat_mean": "TSAT",
    "tsat_latest": "Latest TSAT",
    "log_ferritin": "log(Ferritin)",
    "ferritin_deficiency": "Ferritin Deficiency",
    "tsat_deficiency": "TSAT Deficiency",
    "ferritin_crp_interaction": "Ferritin-CRP Interaction",
    "ferritin_missing": "Ferritin Missing",
    "tsat_missing": "TSAT Missing",
    "delta_ferritin_mean": "Ferritin Change",
    "delta_tsat_mean": "TSAT Change",
    "hif_use_flag": "HIF-PHI Use",
    "current_pre_sbp_mean": "Pre-dialysis SBP",
    "current_pre_dbp_mean": "Pre-dialysis DBP",
    "current_idh_any": "Intradialytic Hypotension",
    "esa_use_flag": "ESA Use",
    "prior_low_response_proxy": "Prior Low Response",
}

# Features to hide from SHAP display
SHAP_HIDE = {
    "center_creator", "receiving_center", "patient_status",
    "esa_use", "esa_type", "iron_use", "hif_use", "esa_unit",
    "sex", "primary_disease", "esa_route",
    "window", "year", "quarter", "patient_id",
    "post_sbp_q1_mean", "post_sbp_q1_std", "post_dbp_q1_mean", "post_dbp_q1_std",
    "dialysis_frequency", "dialysis_hours",
    "pre_sbp_q1_std", "pre_dbp_q1_std",
    "pre_sbp_q2_mean", "pre_dbp_q2_mean", "idh_any_q2",
    "pre_sbp_q3_mean", "pre_dbp_q3_mean", "idh_any_q3",
    "delta_hb", "delta_esa_dose", "delta_eri", "delta_crp",
    "delta_albumin", "delta_ktv", "delta_pth",
    "idh_count_q1", "pre_sbp_q1_mean", "pre_dbp_q1_mean", "idh_any_q1",
}

# ---------------------------------------------------------------------------
# Decision rule definitions
# ---------------------------------------------------------------------------

DECISION_TIERS = {
    "urgent": {
        "label": "URGENT",
        "color": "#c62828",
        "bg_color": "#ffebee",
        "css_class": "tier-urgent",
        "description": "Requires immediate clinical attention",
    },
    "primary": {
        "label": "PRIMARY OPTIMIZATION",
        "color": "#e65100",
        "bg_color": "#fff3e0",
        "css_class": "tier-primary",
        "description": "SHAP-driven, address key risk contributors",
    },
    "phenotype": {
        "label": "PHENOTYPE-ALIGNED",
        "color": "#1565c0",
        "bg_color": "#e3f2fd",
        "css_class": "tier-phenotype",
        "description": "Specific to patient's clinical phenotype",
    },
    "supportive": {
        "label": "SUPPORTIVE",
        "color": "#2e7d32",
        "bg_color": "#e8f5e9",
        "css_class": "tier-supportive",
        "description": "Additional optimization opportunities",
    },
}

# Evidence levels
EVIDENCE_LEVELS = {
    "1A": "Strong recommendation, high-quality evidence",
    "1B": "Strong recommendation, moderate-quality evidence",
    "1C": "Strong recommendation, low-quality evidence",
    "2A": "Weak recommendation, high-quality evidence",
    "2B": "Weak recommendation, moderate-quality evidence",
    "2C": "Weak recommendation, low-quality evidence",
    "Expert": "Expert opinion / clinical experience",
}

# Publication figure settings
FIGURE_DPI = 300
FIGURE_FORMATS = ["png", "pdf"]
FIGURE_FONT_FAMILY = "Arial"
FIGURE_FONT_SIZE = 12
FIGURE_TITLE_SIZE = 14
FIGURE_SIZE_SINGLE = (8, 6)
FIGURE_SIZE_DOUBLE = (16, 8)
FIGURE_SIZE_TRIPLE = (18, 6)
