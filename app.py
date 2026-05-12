"""
ESA Hyporesponsiveness Clinical Decision Support System
========================================================
Complete Streamlit web application with:
  - Individual patient risk prediction & phenotype classification
  - SHAP-based explainability
  - Multi-tier clinical decision recommendations
  - Batch patient evaluation
  - Follow-up tracking
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import html
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime

from modules.inference import predict_case
from modules.config import (
    INPUT_FEATURES, DECISION_TIERS, SUBTYPE_NAMES, SUBTYPE_SHORT_NAMES,
    RISK_COLORS, SUBTYPE_DESCRIPTIONS,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ESA-CDSS | Clinical Decision Support",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; color: #1a237e; }
    .subtitle { font-size: 1rem; color: #546e7a; margin-bottom: 1.5rem; }
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .tier-urgent {
        border-left: 5px solid #c62828;
        background: #ffebee;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
    }
    .tier-primary {
        border-left: 5px solid #e65100;
        background: #fff3e0;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
    }
    .tier-phenotype {
        border-left: 5px solid #1565c0;
        background: #e3f2fd;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
    }
    .tier-supportive {
        border-left: 5px solid #2e7d32;
        background: #e8f5e9;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
    }
    .tier-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 0.3rem; }
    .tier-detail { font-size: 0.92rem; line-height: 1.6; white-space: pre-line; }
    .badge-row { display:flex; gap:0.4rem; flex-wrap:wrap; margin:0.4rem 0 0.6rem 0; }
    .badge { font-size:0.72rem; font-weight:700; padding:0.2rem 0.45rem; border-radius:999px; background:#eceff1; color:#37474f; }
    .severity-emergency { background:#b71c1c; color:white; }
    .severity-high { background:#e65100; color:white; }
    .severity-medium { background:#fdd835; color:#3e2723; }
    .severity-low { background:#c8e6c9; color:#1b5e20; }
    .action-summary {
        background:#f5f7fb; border:1px solid #dfe7f3; border-radius:12px;
        padding:1rem 1.2rem; margin-bottom:1rem;
    }
    .action-block { margin-top:0.45rem; }
    .action-label { font-weight:700; color:#263238; margin-bottom:0.15rem; }
    .action-list { margin:0.1rem 0 0.2rem 1.1rem; padding:0; }
    .action-list li { margin-bottom:0.2rem; line-height:1.45; }
    .avoid-text { color:#8a4b08; }
    .monitor-text { color:#1b5e20; }
    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #1a237e;
        border-bottom: 2px solid #c5cae9;
        padding-bottom: 0.4rem; margin: 1.5rem 0 1rem 0;
    }
    .ref-range { font-size: 0.78rem; color: #78909c; }
    .phenotype-card {
        background: linear-gradient(135deg, #e8eaf6, #c5cae9);
        border-radius: 12px; padding: 1.2rem; text-align: center;
    }
    .phenotype-name { font-size: 1.3rem; font-weight: 700; color: #283593; }
    .disclaimer {
        font-size: 0.82rem; color: #78909c;
        border-top: 1px solid #e0e0e0;
        padding-top: 0.8rem; margin-top: 2rem;
    }
    .metric-value { font-size: 1.8rem; font-weight: 800; }
    .evidence-tag {
        font-size: 0.75rem; color: #546e7a;
        background: #eceff1; padding: 0.2rem 0.5rem;
        border-radius: 4px; display: inline-block; margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _esc(value):
    return html.escape(str(value)) if value is not None else ""


def _render_action_items(items):
    return "".join(f"<li>{_esc(item)}</li>" for item in (items or []))


def _recommendation_sort_key(sg):
    severity_order = {"emergency": 0, "high": 1, "medium": 2, "low": 3}
    timeframe_order = {
        "Immediate / same day": 0,
        "Within 1 week": 1,
        "2-4 weeks": 2,
        "4-8 week reassessment": 3,
    }
    tier_order = {"urgent": 0, "primary": 1, "phenotype": 2, "supportive": 3}
    return (
        severity_order.get(sg.get("severity", "low"), 9),
        timeframe_order.get(sg.get("timeframe", "4-8 week reassessment"), 9),
        tier_order.get(sg.get("tier", "supportive"), 9),
        sg.get("priority", 999),
    )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

page = st.sidebar.selectbox(
    "Navigation",
    ["Individual Patient Assessment", "Batch Evaluation", "About This System"],
)

# ---------------------------------------------------------------------------
# Page: Individual Patient Assessment
# ---------------------------------------------------------------------------

if page == "Individual Patient Assessment":
    with st.sidebar:
        st.header("Patient Data Entry")
        st.caption("Enter current-quarter clinical parameters.")

        # -- Demographics --
        with st.expander("Demographics", expanded=True):
            cfg = INPUT_FEATURES["demographics"]
            age = st.number_input(f"Age ({cfg['age']['unit']})",
                                  min_value=cfg["age"]["min"], max_value=cfg["age"]["max"],
                                  value=cfg["age"]["default"], help=cfg["age"]["help"])
            dialysis_age = st.number_input(f"Dialysis Vintage ({cfg['dialysis_age']['unit']})",
                                           min_value=cfg["dialysis_age"]["min"],
                                           value=cfg["dialysis_age"]["default"],
                                           help=cfg["dialysis_age"]["help"])
            st.markdown(f'<p class="ref-range">Typical range: 0-200+ months</p>',
                        unsafe_allow_html=True)

        # -- Anemia & ESA --
        with st.expander("Anemia & ESA", expanded=True):
            cfg = INPUT_FEATURES["anemia_esa"]
            hb = st.number_input(f"Hemoglobin ({cfg['hb']['unit']})",
                                 min_value=cfg["hb"]["min"], max_value=cfg["hb"]["max"],
                                 value=cfg["hb"]["default"])
            st.markdown(f'<p class="ref-range">{cfg["hb"]["ref"]}</p>', unsafe_allow_html=True)
            esa_dose = st.number_input(f"ESA Weekly Dose ({cfg['esa_dose']['unit']})",
                                       min_value=cfg["esa_dose"]["min"],
                                       value=cfg["esa_dose"]["default"],
                                       step=cfg["esa_dose"].get("step", 1.0))
            esa_route = st.selectbox("ESA Route", cfg["esa_route"]["options"])
            dry_weight = st.number_input(f"Dry Weight ({cfg['dry_weight']['unit']})",
                                         min_value=cfg["dry_weight"]["min"],
                                         max_value=cfg["dry_weight"]["max"],
                                         value=cfg["dry_weight"]["default"])

        # -- Inflammation & Nutrition --
        with st.expander("Inflammation & Nutrition"):
            cfg = INPUT_FEATURES["inflammation_nutrition"]
            crp = st.number_input(f"CRP ({cfg['crp']['unit']})",
                                  min_value=cfg["crp"]["min"], value=cfg["crp"]["default"],
                                  step=cfg["crp"].get("step", 1.0))
            st.markdown(f'<p class="ref-range">{cfg["crp"]["ref"]}</p>', unsafe_allow_html=True)
            albumin = st.number_input(f"Albumin ({cfg['albumin']['unit']})",
                                      min_value=cfg["albumin"]["min"],
                                      max_value=cfg["albumin"]["max"],
                                      value=cfg["albumin"]["default"])
            st.markdown(f'<p class="ref-range">{cfg["albumin"]["ref"]}</p>', unsafe_allow_html=True)

        # -- Iron Status --
        with st.expander("Iron Status", expanded=True):
            cfg = INPUT_FEATURES["iron_status"]
            ferritin_mean = st.number_input(f"Ferritin ({cfg['ferritin_mean']['unit']})",
                                            min_value=cfg["ferritin_mean"]["min"],
                                            max_value=cfg["ferritin_mean"]["max"],
                                            value=cfg["ferritin_mean"]["default"],
                                            step=cfg["ferritin_mean"].get("step", 1.0))
            st.markdown(f'<p class="ref-range">{cfg["ferritin_mean"]["ref"]}</p>', unsafe_allow_html=True)
            tsat_mean = st.number_input(f"TSAT ({cfg['tsat_mean']['unit']})",
                                        min_value=cfg["tsat_mean"]["min"],
                                        max_value=cfg["tsat_mean"]["max"],
                                        value=cfg["tsat_mean"]["default"],
                                        step=cfg["tsat_mean"].get("step", 0.5))
            st.markdown(f'<p class="ref-range">{cfg["tsat_mean"]["ref"]}</p>', unsafe_allow_html=True)

        # -- Dialysis Adequacy --
        with st.expander("Dialysis Adequacy"):
            cfg = INPUT_FEATURES["dialysis_adequacy"]
            ktv = st.number_input("Kt/V", min_value=cfg["ktv"]["min"],
                                  max_value=cfg["ktv"]["max"], value=cfg["ktv"]["default"],
                                  step=cfg["ktv"].get("step", 0.01))
            st.markdown(f'<p class="ref-range">{cfg["ktv"]["ref"]}</p>', unsafe_allow_html=True)
            urr = st.number_input(f"URR ({cfg['urr']['unit']})",
                                  min_value=cfg["urr"]["min"], max_value=cfg["urr"]["max"],
                                  value=cfg["urr"]["default"])
            st.markdown(f'<p class="ref-range">{cfg["urr"]["ref"]}</p>', unsafe_allow_html=True)

        # -- CKD-MBD --
        with st.expander("CKD-MBD"):
            cfg = INPUT_FEATURES["ckd_mbd"]
            pth = st.number_input(f"PTH ({cfg['pth']['unit']})",
                                  min_value=cfg["pth"]["min"], value=cfg["pth"]["default"])
            st.markdown(f'<p class="ref-range">{cfg["pth"]["ref"]}</p>', unsafe_allow_html=True)
            calcium = st.number_input(f"Calcium ({cfg['calcium']['unit']})",
                                      min_value=cfg["calcium"]["min"],
                                      max_value=cfg["calcium"]["max"],
                                      value=cfg["calcium"]["default"],
                                      step=cfg["calcium"].get("step", 0.01))
            st.markdown(f'<p class="ref-range">{cfg["calcium"]["ref"]}</p>', unsafe_allow_html=True)
            phosphorus = st.number_input(f"Phosphorus ({cfg['phosphorus']['unit']})",
                                         min_value=cfg["phosphorus"]["min"],
                                         max_value=cfg["phosphorus"]["max"],
                                         value=cfg["phosphorus"]["default"],
                                         step=cfg["phosphorus"].get("step", 0.01))
            st.markdown(f'<p class="ref-range">{cfg["phosphorus"]["ref"]}</p>', unsafe_allow_html=True)

        # -- Electrolytes --
        with st.expander("Electrolytes"):
            cfg = INPUT_FEATURES["electrolytes"]
            potassium = st.number_input(f"Potassium ({cfg['potassium']['unit']})",
                                        min_value=cfg["potassium"]["min"],
                                        max_value=cfg["potassium"]["max"],
                                        value=cfg["potassium"]["default"],
                                        step=cfg["potassium"].get("step", 0.1))
            sodium = st.number_input(f"Sodium ({cfg['sodium']['unit']})",
                                     min_value=cfg["sodium"]["min"],
                                     max_value=cfg["sodium"]["max"],
                                     value=cfg["sodium"]["default"])
            creatinine = st.number_input(f"Creatinine ({cfg['creatinine']['unit']})",
                                         min_value=cfg["creatinine"]["min"],
                                         value=cfg["creatinine"]["default"])

        # -- Hemodynamics --
        with st.expander("Hemodynamics"):
            cfg = INPUT_FEATURES["hemodynamics"]
            sbp = st.number_input(f"Pre-dialysis SBP ({cfg['sbp']['unit']})",
                                  min_value=cfg["sbp"]["min"], max_value=cfg["sbp"]["max"],
                                  value=cfg["sbp"]["default"])
            dbp = st.number_input(f"Pre-dialysis DBP ({cfg['dbp']['unit']})",
                                  min_value=cfg["dbp"]["min"], max_value=cfg["dbp"]["max"],
                                  value=cfg["dbp"]["default"])
            idh_any = st.checkbox("Intradialytic Hypotension (IDH)", value=False)

        # -- Medications --
        with st.expander("Medications"):
            iron_use = st.checkbox("Iron Supplement", value=True)
            hif_use = st.checkbox("HIF-PHI", value=False)

        submitted = st.button("Analyze Patient", type="primary", use_container_width=True)

    # Header
    st.markdown('<p class="main-title">ESA Hyporesponsiveness Clinical Decision Support</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">'
        'Individualized risk prediction, phenotype classification, and evidence-based '
        'treatment recommendations for ESA low-response in maintenance hemodialysis patients.'
        '</p>', unsafe_allow_html=True
    )

    if not submitted:
        st.info("Enter patient data in the sidebar and click **Analyze Patient** to begin.")
        st.stop()

    # Run prediction
    with st.spinner("Running risk prediction, phenotype classification, and generating recommendations..."):
        values = {
            "age": age, "dialysis_age": dialysis_age,
            "hb": hb, "esa_dose": esa_dose, "esa_route": esa_route,
            "dry_weight": dry_weight, "crp": crp, "albumin": albumin,
            "ferritin_mean": ferritin_mean, "tsat_mean": tsat_mean,
            "ktv": ktv, "urr": urr, "pth": pth,
            "calcium": calcium, "phosphorus": phosphorus,
            "potassium": potassium, "sodium": sodium, "creatinine": creatinine,
            "iron_use_flag": 1 if iron_use else 0,
            "hif_use_flag": 1 if hif_use else 0,
            "current_pre_sbp_mean": sbp,
            "current_pre_dbp_mean": dbp,
            "current_idh_any": 1 if idh_any else 0,
            "pre_sbp_q1_mean": sbp, "pre_dbp_q1_mean": dbp,
            "idh_any_q1": 1 if idh_any else 0,
        }
        result = predict_case(values)

    prob = result["risk_score"]
    risk_level = result["risk_level"]
    risk_color = result["risk_color"]
    risk_desc = result["risk_description"]
    phenotype = result["phenotype"]
    phenotype_short = result["phenotype_short"]
    phenotype_desc = result["phenotype_description"]
    distances = result["phenotype_distances"]
    shap_data = result["shap"]
    suggestions = result["suggestions"]

    # ================================================================
    # Section 1: Risk Assessment
    # ================================================================

    st.markdown('<p class="section-header">1. Risk Assessment</p>', unsafe_allow_html=True)

    col_gauge, col_info = st.columns([1, 1.2])

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(prob * 100, 1),
            domain={"x": [0, 1], "y": [0, 1]},
            number={"suffix": "%", "font": {"size": 36, "color": risk_color}},
            delta={"reference": 20, "decreasing": {"color": "#27ae60"},
                   "increasing": {"color": "#e74c3c"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#455a64"},
                "bar": {"color": risk_color, "thickness": 0.35},
                "bgcolor": "white", "borderwidth": 1, "bordercolor": "#cfd8dc",
                "steps": [
                    {"range": [0, 20], "color": "#e8f5e9"},
                    {"range": [20, 50], "color": "#fff8e1"},
                    {"range": [50, 80], "color": "#ffebee"},
                    {"range": [80, 100], "color": "#f3e5f5"},
                ],
                "threshold": {
                    "line": {"color": "#37474f", "width": 3},
                    "thickness": 0.8, "value": 20,
                },
            },
            title={"text": "ESA Hyporesponsiveness Risk", "font": {"size": 16}},
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=50, b=20, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_info:
        st.markdown(f"""
        <div class="result-card">
            <h3 style="margin:0 0 0.5rem 0;">Predicted Risk Level</h3>
            <p style="font-size:2rem; font-weight:800; color:{risk_color}; margin:0;">
                {risk_level}
            </p>
            <p style="color:#455a64; margin:0.5rem 0 0 0;">{risk_desc}</p>
        </div>
        """, unsafe_allow_html=True)

        eq_dose = esa_dose if esa_route == "Subcutaneous" else esa_dose * 2 / 3
        eri_val = eq_dose / dry_weight / (hb / 10.0) if hb > 0 and dry_weight > 0 else 0
        st.markdown(f"""
        <div class="result-card">
            <p style="color:#546e7a; margin:0;">
                <b>ERI:</b> {eri_val:.1f} &nbsp;&nbsp;
                <b>Ferritin:</b> {ferritin_mean:.0f} ng/mL &nbsp;&nbsp;
                <b>TSAT:</b> {tsat_mean:.1f}% &nbsp;&nbsp;
                <b>ESA Route:</b> {esa_route} &nbsp;&nbsp;
                <b>Eq. Dose:</b> {eq_dose:.0f} IU/week
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ================================================================
    # Section 2: Phenotype Classification
    # ================================================================

    st.markdown('<p class="section-header">2. Phenotype Classification</p>', unsafe_allow_html=True)

    pheno_col1, pheno_col2 = st.columns([1, 1.5])

    with pheno_col1:
        st.markdown(f"""
        <div class="phenotype-card">
            <p style="font-size:0.85rem; color:#546e7a; margin:0 0 0.3rem 0;">
                Assigned Subtype
            </p>
            <p class="phenotype-name">{phenotype_short}</p>
            <p style="font-size:0.8rem; color:#546e7a; margin:0.5rem 0 0 0; text-align:left;">
                {phenotype_desc[:200]}...
            </p>
        </div>
        """, unsafe_allow_html=True)

    with pheno_col2:
        subtype_names_list = list(distances.keys())
        similarities = [distances[n]["similarity"] for n in subtype_names_list]

        fig_pheno = go.Figure(go.Bar(
            x=similarities,
            y=subtype_names_list,
            orientation="h",
            marker_color=["#1e88e5" if n == phenotype else "#90caf9"
                          for n in subtype_names_list],
            text=[f"{s:.1%}" for s in similarities],
            textposition="outside",
        ))
        fig_pheno.update_layout(
            title="Similarity to Each Subtype",
            xaxis_title="Similarity Score",
            xaxis_range=[0, 1.15],
            height=200,
            margin=dict(t=40, b=30, l=20, r=60),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_pheno, use_container_width=True)

    # ================================================================
    # Section 3: Risk Factor Analysis (SHAP)
    # ================================================================

    st.markdown('<p class="section-header">3. Individualized Risk Factor Analysis</p>',
                unsafe_allow_html=True)

    top_n = min(10, len(shap_data["drivers"]))
    if top_n > 0:
        drivers = shap_data["drivers"][:top_n]
        labels = [d["label"] for d in drivers]
        shap_vals = [d["shap"] for d in drivers]
        colors = ["#e53935" if v > 0 else "#1e88e5" for v in shap_vals]

        fig_shap = go.Figure(go.Bar(
            x=shap_vals, y=labels, orientation="h",
            marker_color=colors,
            text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in shap_vals],
            textposition="outside",
        ))
        fig_shap.update_layout(
            title="Top Risk-Increasing Factors (SHAP Values)",
            xaxis_title="SHAP Value (impact on prediction)",
            height=max(300, top_n * 35 + 60),
            margin=dict(t=40, b=30, l=20, r=60),
            yaxis=dict(autorange="reversed"),
        )
        fig_shap.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")
        st.plotly_chart(fig_shap, use_container_width=True)

        if shap_data["protective"]:
            st.markdown("**Protective Factors** (reducing risk):")
            prot_text = " | ".join(
                f"{p['label']} (SHAP {p['shap']:.3f})"
                for p in shap_data["protective"][:5]
            )
            st.markdown(f'<div class="result-card"><p style="color:#2e7d32; margin:0;">{prot_text}</p></div>',
                        unsafe_allow_html=True)
    else:
        st.info("No significant risk drivers identified from SHAP analysis.")

    # ================================================================
    # Section 4: Clinical Recommendations
    # ================================================================

    st.markdown('<p class="section-header">4. Clinical Decision Recommendations</p>',
                unsafe_allow_html=True)

    tier_css = {
        "urgent": "tier-urgent", "primary": "tier-primary",
        "phenotype": "tier-phenotype", "supportive": "tier-supportive",
    }
    tier_labels = {
        "urgent": "URGENT", "primary": "PRIMARY OPTIMIZATION",
        "phenotype": "PHENOTYPE-ALIGNED", "supportive": "SUPPORTIVE",
    }

    sorted_suggestions = sorted(suggestions, key=_recommendation_sort_key)
    top_three = sorted_suggestions[:3]
    avoid_items = [
        sg for sg in sorted_suggestions
        if sg.get("avoid") and not str(sg.get("avoid", "")).startswith("None specific")
    ][:3]

    top_html = "".join(
        f"<li><b>{_esc(sg.get('title'))}</b> "
        f"<span class='badge severity-{_esc(sg.get('severity', 'low'))}'>{_esc(sg.get('severity', '').upper())}</span> "
        f"<span class='badge'>{_esc(sg.get('timeframe'))}</span><br>"
        f"<span>{_esc(sg.get('rationale'))}</span></li>"
        for sg in top_three
    )
    avoid_html = "".join(
        f"<li><b>{_esc(sg.get('title'))}:</b> {_esc(sg.get('avoid'))}</li>"
        for sg in avoid_items
    ) or "<li>No specific avoid item beyond standard contraindication review.</li>"

    st.markdown(f"""
    <div class="action-summary">
        <div class="action-label">Top 3 priorities for this patient</div>
        <ol class="action-list">{top_html}</ol>
        <div class="action-label" style="margin-top:0.7rem;">What not to do immediately</div>
        <ul class="action-list avoid-text">{avoid_html}</ul>
    </div>
    """, unsafe_allow_html=True)

    current_tier = None
    for sg in sorted_suggestions:
        tier = sg.get("tier", "supportive")
        if tier != current_tier:
            current_tier = tier
            st.markdown(f"**{tier_labels.get(tier, tier.upper())}**")

        severity = sg.get("severity", "low")
        evidence_html = ""
        if sg.get("evidence"):
            evidence_html = f'<div class="evidence-tag">Evidence: {_esc(sg["evidence"])}</div>'

        actions_html = _render_action_items(sg.get("actions", []))
        st.markdown(f"""
        <div class="{tier_css.get(tier, 'tier-supportive')}">
            <p class="tier-title">{_esc(sg.get("title"))}</p>
            <div class="badge-row">
                <span class="badge">{_esc(tier_labels.get(tier, tier.upper()))}</span>
                <span class="badge severity-{_esc(severity)}">{_esc(severity.upper())}</span>
                <span class="badge">{_esc(sg.get("timeframe"))}</span>
            </div>
            <div class="action-block">
                <div class="action-label">Why this is a priority for this patient</div>
                <div class="tier-detail">{_esc(sg.get("rationale"))}</div>
            </div>
            <div class="action-block">
                <div class="action-label">Recommended actions</div>
                <ul class="action-list">{actions_html}</ul>
            </div>
            <div class="action-block avoid-text">
                <div class="action-label">Avoid / use caution</div>
                <div class="tier-detail">{_esc(sg.get("avoid"))}</div>
            </div>
            <div class="action-block monitor-text">
                <div class="action-label">Monitoring plan</div>
                <div class="tier-detail">{_esc(sg.get("monitoring"))}</div>
            </div>
            {evidence_html}
        </div>
        """, unsafe_allow_html=True)

    # ================================================================
    # Disclaimer
    # ================================================================

    st.markdown("""
    <div class="disclaimer">
        <b>Disclaimer:</b> This tool is designed as a clinical decision support aid only.
        It does not replace clinical judgment. Predictions are based on a statistical model
        trained on retrospective data and may not generalize to all clinical scenarios.
        Always correlate with the patient's full clinical picture and current guidelines.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page: Batch Evaluation
# ---------------------------------------------------------------------------

elif page == "Batch Evaluation":
    st.markdown('<p class="main-title">Batch Patient Evaluation</p>', unsafe_allow_html=True)
    st.markdown("Upload a CSV file with patient data to run batch risk prediction and phenotype classification.")

    st.markdown("""
    **Required columns:** age, dialysis_age, hb, esa_dose, esa_route (Subcutaneous/Intravenous),
    dry_weight, crp, albumin, ferritin_mean, tsat_mean, ktv, urr, pth, calcium, phosphorus,
    potassium, sodium, creatinine, current_pre_sbp_mean, current_pre_dbp_mean,
    current_idh_any (0/1), iron_use_flag (0/1), hif_use_flag (0/1)
    """)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df)} patients.", df.head())

        if st.button("Run Batch Prediction"):
            progress = st.progress(0)
            results = []

            for i, (_, row) in enumerate(df.iterrows()):
                vals = row.to_dict()
                try:
                    result = predict_case(vals)
                    suggestions_result = result["suggestions"]
                    sorted_result_suggestions = sorted(suggestions_result, key=_recommendation_sort_key)
                    top = sorted_result_suggestions[0] if sorted_result_suggestions else {}
                    urgent_count = len([s for s in suggestions_result if s.get("severity") == "emergency" or s.get("tier") == "urgent"])
                    high_priority_count = len([s for s in suggestions_result if s.get("severity") in ("emergency", "high")])
                    top_titles = [s.get("title", "") for s in sorted_result_suggestions[:3]]
                    top_timeframes = [s.get("timeframe", "") for s in sorted_result_suggestions[:3]]
                    results.append({
                        "index": i,
                        "risk_score": result["risk_score"],
                        "risk_level": result["risk_level"],
                        "phenotype": result["phenotype_short"],
                        "top_priority_title": top.get("title", ""),
                        "top_priority_timeframe": top.get("timeframe", ""),
                        "top_priority_reason": top.get("rationale", ""),
                        "urgent_count": urgent_count,
                        "high_priority_count": high_priority_count,
                        "top_3_titles": " | ".join(top_titles),
                        "top_3_timeframes": " | ".join(top_timeframes),
                        "n_suggestions": len(suggestions_result),
                    })
                except Exception as e:
                    results.append({
                        "index": i,
                        "risk_score": None,
                        "risk_level": "Error",
                        "phenotype": str(e)[:50],
                        "top_priority_title": "Prediction failed",
                        "top_priority_timeframe": "",
                        "top_priority_reason": str(e)[:160],
                        "urgent_count": 0,
                        "high_priority_count": 0,
                        "top_3_titles": "",
                        "top_3_timeframes": "",
                        "n_suggestions": 0,
                    })
                progress.progress((i + 1) / len(df))

            results_df = pd.DataFrame(results)
            if not results_df.empty:
                risk_rank = {"Very High": 3, "High": 2, "Intermediate": 1, "Low": 0, "Error": -1}
                results_df["risk_rank"] = results_df["risk_level"].map(risk_rank).fillna(-1)
                results_df = results_df.sort_values(
                    ["urgent_count", "high_priority_count", "risk_rank", "risk_score"],
                    ascending=[False, False, False, False],
                    na_position="last",
                ).drop(columns=["risk_rank"])
            st.session_state["batch_results"] = results_df

        if "batch_results" in st.session_state:
            res = st.session_state["batch_results"]
            st.markdown("### Batch Results Summary")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Patients", len(res))
            col2.metric("Urgent Queue", int(res.get("urgent_count", pd.Series(dtype=int)).sum()))
            col3.metric("High-Priority Items", int(res.get("high_priority_count", pd.Series(dtype=int)).sum()))
            col4.metric("High/Very High Risk",
                        len(res[res["risk_level"].isin(["High", "Very High"])]))
            col5.metric("Low Risk", len(res[res["risk_level"] == "Low"]))

            # Risk distribution
            risk_counts = res["risk_level"].value_counts()
            fig_risk = px.pie(values=risk_counts.values, names=risk_counts.index,
                              title="Risk Level Distribution",
                              color=risk_counts.index,
                              color_discrete_map=RISK_COLORS)
            st.plotly_chart(fig_risk, use_container_width=True)

            # Phenotype distribution
            pheno_counts = res["phenotype"].value_counts()
            fig_pheno_batch = px.bar(x=pheno_counts.index, y=pheno_counts.values,
                                     title="Phenotype Distribution",
                                     labels={"x": "Phenotype", "y": "Count"})
            st.plotly_chart(fig_pheno_batch, use_container_width=True)

            st.markdown("### Ward Round / Quality-Control Priority Queue")
            priority_cols = [
                "index", "risk_score", "risk_level", "phenotype", "urgent_count",
                "high_priority_count", "top_priority_title", "top_priority_timeframe",
                "top_priority_reason", "top_3_titles", "top_3_timeframes", "n_suggestions",
            ]
            display_cols = [c for c in priority_cols if c in res.columns]
            st.dataframe(res[display_cols], use_container_width=True)

            csv = res.to_csv(index=False)
            st.download_button("Download Results CSV", csv, "batch_results.csv", "text/csv")

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------

elif page == "About This System":
    st.markdown('<p class="main-title">About This System</p>', unsafe_allow_html=True)

    st.markdown("""
    ### ESA Hyporesponsiveness Clinical Decision Support System (CDSS)

    This system integrates machine learning-based risk prediction, phenotype classification,
    and evidence-based clinical recommendations to assist nephrologists in managing
    ESA-hyporesponsive hemodialysis patients.

    #### System Architecture
    - **Prediction Model:** XGBoost classifier updated with ferritin and TSAT-derived iron-status features (trained on 3,726 patients, externally validated on 5,114)
    - **Phenotype Classification:** K-Means clustering (K=3, 19 clinical variables including log-ferritin and TSAT, stability ARI median = 0.709)
    - **Explainability:** SHAP (SHapley Additive exPlanations) for individual-level risk attribution
    - **Decision Engine:** Multi-tier recommendation engine integrating KDIGO/KDOQI guidelines

    #### Three Phenotypes
    1. **Hemodynamic instability** - Intradialytic hypotension and blood pressure instability
    2. **MBD-dominant** - Mineral bone disorder and secondary hyperparathyroidism burden
    3. **inflammation-malnutrition-underdialysis** - Anemia burden, inflammation, dialysis adequacy, and iron restriction

    #### Recommendation Tiers
    | Tier | Description |
    |------|-------------|
    | URGENT | Requires immediate clinical attention |
    | PRIMARY OPTIMIZATION | SHAP-driven, address key risk contributors |
    | PHENOTYPE-ALIGNED | Specific to patient's clinical phenotype |
    | SUPPORTIVE | Additional optimization opportunities |

    #### Disclaimer
    This tool is designed as a clinical decision support aid only. It does not replace
    clinical judgment. Always correlate with the patient's full clinical picture.
    """)
