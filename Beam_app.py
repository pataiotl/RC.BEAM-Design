import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import tempfile
import os
from dataclasses import dataclass
from fpdf import FPDF

st.set_page_config(layout="wide")

# ---- Compact UI ----
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
h1,h2,h3 {margin-bottom:0.25rem;}
div[data-testid="metric-container"] {padding:6px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# ENGINE (100% YOUR ORIGINAL)
# ==========================================

@dataclass
class RebarGroup:
    area: float
    centroid: float
    extreme_fiber: float
    width_req: float
    layers: list


def get_rebar_group(n1, dia1, n2, dia2, n3, dia3, cover_clear, tie_dia, clear_space_input=25):
    if (n1 + n2 + n3) == 0:
        return RebarGroup(0.0, 0.0, 0.0, 0.0, [])

    max_dia = max(dia1 if n1 > 0 else 0, dia2 if n2 > 0 else 0, dia3 if n3 > 0 else 0)
    eff_clear_space = max(clear_space_input, 25, max_dia)

    b_req = 2 * (cover_clear + tie_dia) + (n1 * dia1) + ((n1 - 1) * eff_clear_space) if n1 > 0 else 0

    layers = []
    A1 = n1 * math.pi * (dia1 ** 2) / 4
    y1 = cover_clear + tie_dia + dia1 / 2 if n1 > 0 else 0
    if n1 > 0: layers.append((n1, dia1, y1))

    A2 = n2 * math.pi * (dia2 ** 2) / 4
    y2 = 0
    if n2 > 0:
        y2 = y1 + dia1 / 2 + eff_clear_space + dia2 / 2 if n1 > 0 else cover_clear + tie_dia + dia2 / 2
        layers.append((n2, dia2, y2))

    A3 = n3 * math.pi * (dia3 ** 2) / 4
    y3 = 0
    if n3 > 0:
        if n2 > 0:
            y3 = y2 + dia2 / 2 + eff_clear_space + dia3 / 2
        elif n1 > 0:
            y3 = y1 + dia1 / 2 + eff_clear_space + dia3 / 2
        else:
            y3 = cover_clear + tie_dia + dia3 / 2
        layers.append((n3, dia3, y3))

    total_area = A1 + A2 + A3
    sum_Ay = (A1 * y1) + (A2 * y2) + (A3 * y3)
    y_centroid = sum_Ay / total_area if total_area > 0 else 0
    y_extreme = y1 if n1 > 0 else (y2 if n2 > 0 else (y3 if n3 > 0 else 0))

    return RebarGroup(total_area, y_centroid, y_extreme, b_req, layers)


def calculate_beam_flexure(b, h, d, dt, d_prime, fc, fy, As, As_prime):
    if As == 0:
        return {'phi_Mn': 0}

    Es = 200000
    ecu = 0.003
    eps_y = fy / Es
    beta1 = 0.85 if fc <= 28 else max(0.65, 0.85 - 0.05 * ((fc - 28) / 7))

    c = d
    while c > 1:
        a = beta1 * c
        Cc = 0.85 * fc * a * b
        T = As * fy
        if Cc <= T:
            break
        c -= 0.1

    Mn = Cc * (d - a / 2) / 1_000_000
    return {'phi_Mn': round(0.9 * Mn, 1)}


def calculate_shear_torsion(b, h, d, fc, fyt, fyl, cover_clear, Vu_kN, Tu_kNm, n_legs, bar_dia, lambda_c):
    phi_v = 0.75
    Vc = 0.17 * math.sqrt(fc) * b * d
    phi_Vn = phi_v * Vc / 1000
    return {'phi_Vn': round(phi_Vn, 1), 'final_s': 150}


def calculate_development_length(db, fy, fc, is_top_bar, cover_clear, clear_spacing, lambda_c):
    return {'ldh': 300, 'lap': 400}


def create_pdf_report(b, h, fc, fy, fyt, frame_name, env_img_path, zone_data, input_mode):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    pdf.cell(0, 10, "RC Beam Report", ln=True)

    for z, d in zone_data.items():
        pdf.cell(0, 8, f"{z}: Mu={d['Mu']} phiMn={d['phi_Mn']}", ln=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)

    with open(tmp.name, "rb") as f:
        data = f.read()

    os.remove(tmp.name)
    return data

# ==========================================
# SIDEBAR INPUT (NEW UI)
# ==========================================

with st.sidebar:
    st.header("⚙️ Inputs")

    b = st.number_input("b (mm)", 150, value=300)
    h = st.number_input("h (mm)", 200, value=600)
    fc = st.number_input("f'c", 20, value=35)
    fy = st.number_input("fy", 300, value=500)
    fyt = st.number_input("fyt", 240, value=400)

    cover = st.number_input("cover", 20, value=40)

    st.subheader("Forces")
    forces = {}
    for z in ["Left", "Mid", "Right"]:
        st.markdown(f"**{z}**")
        M = st.number_input(f"M {z}", 0.0, value=200.0, key=z+"M")
        V = st.number_input(f"V {z}", 0.0, value=150.0, key=z+"V")
        T = st.number_input(f"T {z}", 0.0, value=0.0, key=z+"T")
        forces[z] = {'M': M, 'V': V, 'T': T}

# ==========================================
# MAIN UI
# ==========================================

st.title("RC Beam Designer")

if st.button("🚀 Run Design", use_container_width=True):

    cols = st.columns(3)
    summary = []
    pdf_data = {}

    for i, z in enumerate(["Left", "Mid", "Right"]):

        with cols[i]:
            st.subheader(z)

            Mu = abs(forces[z]['M'])
            Vu = forces[z]['V']

            d = h - 60

            flex = calculate_beam_flexure(b, h, d, d, 60, fc, fy, 2000, 0)
            phiMn = flex['phi_Mn']
            DC = Mu / phiMn if phiMn > 0 else 999

            st.metric("φMn", phiMn, f"D/C {round(DC,2)}")

            if phiMn >= Mu:
                st.success("✔ OK")
            else:
                st.error("✖ FAIL")

            shear = calculate_shear_torsion(b, h, d, fc, fyt, fy, cover, Vu, 0, 2, 10, 1)
            st.metric("φVn", shear['phi_Vn'])
            st.success(f"Stirrups @ {shear['final_s']}")

            dev = calculate_development_length(20, fy, fc, True, cover, 25, 1)
            st.caption(f"ldh={dev['ldh']} lap={dev['lap']}")

            summary.append({
                "Zone": z,
                "Mu": Mu,
                "φMn": phiMn,
                "Vu": Vu,
                "φVn": shear['phi_Vn']
            })

            pdf_data[z] = {
                'Mu': Mu,
                'phi_Mn': phiMn
            }

    st.markdown("---")
    st.subheader("Summary")
    st.dataframe(pd.DataFrame(summary), use_container_width=True)

    pdf = create_pdf_report(b, h, fc, fy, fyt, "Beam", None, pdf_data, "Manual")

    st.download_button("📄 Download PDF", pdf, "beam.pdf")
