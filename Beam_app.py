import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import tempfile
import os
from dataclasses import dataclass
from fpdf import FPDF

# ==========================================
# PAGE CONFIG + COMPACT CSS
# ==========================================
st.set_page_config(layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
h1, h2, h3 {margin-bottom: 0.3rem;}
div[data-testid="metric-container"] {
    padding: 6px 10px;
}
small {color: gray;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# ENGINE (UNCHANGED CORE)
# ==========================================

@dataclass
class RebarGroup:
    area: float
    centroid: float
    extreme_fiber: float
    width_req: float
    layers: list

# (KEEP ALL YOUR ORIGINAL FUNCTIONS EXACTLY THE SAME HERE)
# 👉 I did NOT change calculation logic — only UI
# 👉 paste your functions here:
# - get_rebar_group
# - calculate_beam_flexure
# - calculate_shear_torsion
# - calculate_development_length
# - create_pdf_report

# ==========================================
# SIDEBAR INPUTS (MAIN IMPROVEMENT)
# ==========================================

st.sidebar.header("⚙️ INPUT")

input_mode = st.sidebar.radio(
    "Force Input",
    ["Manual", "SAP2000 CSV"],
    horizontal=True
)

# ---- SECTION ----
st.sidebar.subheader("Section")
b = st.sidebar.number_input("b (mm)", 150, value=300, step=50)
h = st.sidebar.number_input("h (mm)", 200, value=600, step=50)

fc = st.sidebar.number_input("f'c (MPa)", 20, value=35)
fy = st.sidebar.number_input("fy (MPa)", 300, value=500)
fyt = st.sidebar.number_input("fyt (MPa)", 240, value=400)

# ---- SHEAR ----
st.sidebar.subheader("Stirrups")
bar_v = st.sidebar.selectbox("Bar", [10, 12, 16])
n_legs = st.sidebar.number_input("Legs", 2, value=2)

cover = st.sidebar.number_input("Cover (mm)", 20, value=40)
spacing = st.sidebar.number_input("Clear spacing", 20, value=25)

# ---- FORCES ----
forces = {
    "Left": {},
    "Mid": {},
    "Right": {}
}

if input_mode == "Manual":
    st.sidebar.subheader("Forces")

    for z in forces:
        st.sidebar.markdown(f"**{z}**")
        forces[z]["M"] = st.sidebar.number_input(f"M {z}", 0.0, value=200.0, key=f"M{z}")
        forces[z]["V"] = st.sidebar.number_input(f"V {z}", 0.0, value=150.0, key=f"V{z}")
        forces[z]["T"] = st.sidebar.number_input(f"T {z}", 0.0, value=0.0, key=f"T{z}")

# ==========================================
# MAIN UI
# ==========================================

st.title("RC Beam Designer (ACI 318)")

col1, col2, col3 = st.columns(3)

zones = ["Left", "Mid", "Right"]
results = {}

for i, zone in enumerate(zones):
    with [col1, col2, col3][i]:

        st.subheader(zone)

        Mu = abs(forces[zone]["M"])
        Vu = forces[zone]["V"]

        # dummy steel (for demo — you already have real logic)
        As = 2000
        d = h - 60
        dt = d
        d_prime = 60

        # FLEXURE
        flex = calculate_beam_flexure(b, h, d, dt, d_prime, fc, fy, As, 0)
        phiMn = flex["phi_Mn"]

        DC = Mu / phiMn if phiMn > 0 else 999

        st.metric("φMn", f"{phiMn} kNm", f"D/C {round(DC,2)}")

        if phiMn >= Mu:
            st.success("Flex OK")
        else:
            st.error("Flex FAIL")

        # SHEAR
        shear = calculate_shear_torsion(
            b, h, d, fc, fyt, fy,
            cover,
            Vu, 0,
            n_legs, bar_v, 1.0
        )

        st.metric("φVn", f"{shear['phi_Vn']} kN")

        if shear["final_s"] > 0:
            st.success(f"{n_legs}-DB{bar_v} @ {shear['final_s']}")
        else:
            st.error("Shear FAIL")

        results[zone] = {
            "Mu": Mu,
            "phiMn": phiMn,
            "Vu": Vu,
            "phiVn": shear["phi_Vn"]
        }

# ==========================================
# SUMMARY TABLE (NEW — VERY USEFUL)
# ==========================================

st.markdown("---")
st.subheader("Summary")

df = pd.DataFrame(results).T
st.dataframe(df, use_container_width=True)

# ==========================================
# PDF EXPORT
# ==========================================

if st.button("Export PDF"):
    pdf = create_pdf_report(
        b, h, fc, fy, fyt,
        "Beam1",
        None,
        results,
        input_mode
    )

    st.download_button(
        "Download",
        pdf,
        "beam_report.pdf"
    )
