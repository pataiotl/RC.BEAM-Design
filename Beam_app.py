import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import tempfile
import os
from dataclasses import dataclass
from fpdf import FPDF

st.set_page_config(layout="wide")


# ==========================================
# 1. THE ENGINEERING ENGINES
# ==========================================

@dataclass
class RebarGroup:
    area: float
    centroid: float
    extreme_fiber: float
    width_req: float
    layers: list


def get_rebar_group(n1, dia1, n2, dia2, n3, dia3, cover_clear, tie_dia, clear_space_input=25) -> RebarGroup:
    if (n1 + n2 + n3) == 0:
        return RebarGroup(0.0, 0.0, 0.0, 0.0, [])

    max_dia = max(dia1 if n1 > 0 else 0, dia2 if n2 > 0 else 0, dia3 if n3 > 0 else 0)
    eff_clear_space = max(clear_space_input, 25, max_dia)

    # Check Layer 1 fit (widest — usually governs)
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
        return {'phi_Mn': 0, 'passes_As_min': False, 'is_ductile': False, 'converged': True,
                'As_min': 0, 'c': 0, 'a': 0, 'eps_t': 0, 'phi': 0, 'Mn': 0}
    Es = 200000
    ecu = 0.003
    eps_y = fy / Es
    beta1 = 0.85 if fc <= 28 else max(0.65, 0.85 - 0.05 * ((fc - 28) / 7))
    As_min = max((0.25 * math.sqrt(fc) / fy) * b * d, (1.4 / fy) * b * d)

    c = d
    converged = False
    while c > 1.0:
        a = min(beta1 * c, h)
        Cc = 0.85 * fc * a * b
        eps_s_prime = ecu * (c - d_prime) / c if c > 0 else 0
        fs_prime = min(fy, max(-fy, eps_s_prime * Es))
        Cs = As_prime * fs_prime
        if d_prime <= a and eps_s_prime > 0:
            Cs -= As_prime * 0.85 * fc
        eps_s = ecu * (d - c) / c if c > 0 else 0
        fs = min(fy, max(-fy, eps_s * Es))
        T = As * fs
        if (Cc + Cs) <= T:
            converged = True
            break
        c -= 0.1

    Mn_Nmm = Cc * (d - a / 2) + Cs * (d - d_prime)
    Mn_kNm = Mn_Nmm / 1_000_000
    eps_t = ecu * (dt - c) / c if c > 0 else 0

    if eps_t <= eps_y:
        phi = 0.65
    elif eps_t >= (eps_y + 0.003):
        phi = 0.90
    else:
        phi = 0.65 + 0.25 * ((eps_t - eps_y) / 0.003)

    return {
        'phi_Mn': round(phi * Mn_kNm, 1), 'is_ductile': eps_t >= 0.004,
        'As_min': round(As_min, 1), 'passes_As_min': As >= As_min,
        'converged': converged, 'c': round(c, 2), 'a': round(a, 2),
        'eps_t': round(eps_t, 5), 'phi': round(phi, 3), 'Mn': round(Mn_kNm, 1)
    }


def calculate_shear_torsion(b, h, d, fc, fyt, fyl, cover_clear, Vu_kN, Tu_kNm, n_legs, bar_dia, lambda_c):
    if d <= 0:
        return {'final_s': 0, 'section_fails': True, 'needs_torsion': False, 'Al_req': 0, 'Al_min': 0,
                's_exact': 0, 's_max': 0, 'combined_stress': 0, 'stress_limit': 0,
                'lambda_s': 1.0, 'phi_Vc': 0, 'phi_Vn': 0, 'T_th': 0}
    phi_v = 0.75
    Vu = abs(Vu_kN) * 1000
    Tu = abs(Tu_kNm) * 1_000_000
    A_leg = math.pi * (bar_dia ** 2) / 4
    lambda_s = min(1.0, math.sqrt(2 / (1 + 0.004 * d)))

    Vc = 0.17 * lambda_s * lambda_c * math.sqrt(fc) * b * d
    phi_Vc = phi_v * Vc

    x1 = b - 2 * (cover_clear + bar_dia / 2)
    y1 = h - 2 * (cover_clear + bar_dia / 2)
    Aoh = x1 * y1
    Ao = 0.85 * Aoh
    ph = 2 * (x1 + y1)
    Acp = b * h
    pcp = 2 * (b + h)

    T_th = phi_v * 0.083 * lambda_c * math.sqrt(fc) * (Acp ** 2 / pcp)
    needs_torsion = Tu > T_th

    Vs_req = max((Vu / phi_v) - Vc, 0) if Vu > phi_Vc / 2 else 0
    Av_s_req = Vs_req / (fyt * d) if Vs_req > 0 else 0

    if needs_torsion:
        At_s_req = (Tu / phi_v) / (2 * Ao * fyt)
        At_s_for_min = max(At_s_req, 0.175 * b / fyt)
        Al_min = max(0, (0.42 * math.sqrt(fc) * Acp / fyl) - (At_s_for_min * ph * (fyt / fyl)))
        Al_req = max(At_s_req * ph * (fyt / fyl), Al_min)
    else:
        At_s_req, Al_req, Al_min = 0, 0, 0

    req_per_outer_leg = (Av_s_req / n_legs) + At_s_req
    s_calc = A_leg / req_per_outer_leg if req_per_outer_leg > 0 else 9999

    min_combined_ratio = max(0.062 * math.sqrt(fc) * b / fyt, 0.35 * b / fyt)
    req_per_leg_min = min_combined_ratio / 2
    s_min_steel = A_leg / req_per_leg_min if req_per_leg_min > 0 else 9999

    s_req = min(s_calc, s_min_steel)
    s_max_shear = min(d / 4, 300) if Vs_req > (0.33 * math.sqrt(fc) * b * d) else min(d / 2, 600)
    s_max = min(s_max_shear, min(ph / 8, 300)) if needs_torsion else s_max_shear

    s_exact = min(s_req, s_max)
    final_s = math.floor(s_exact / 25) * 25

    Vs_prov = (n_legs * A_leg * fyt * d / final_s) if final_s > 0 else 0
    phi_Vn = phi_v * (Vc + Vs_prov)

    v_stress = Vu / (b * d)
    t_stress = (Tu * ph) / (1.7 * (Aoh ** 2)) if needs_torsion else 0
    combined_stress = math.sqrt(v_stress ** 2 + t_stress ** 2)
    stress_limit = phi_v * ((Vc / (b * d)) + 0.66 * math.sqrt(fc))
    section_fails = combined_stress > stress_limit

    return {
        'final_s': 0 if final_s < 50 or section_fails else final_s,
        'section_fails': section_fails, 'needs_torsion': needs_torsion,
        'Al_req': round(Al_req, 1), 'Al_min': round(Al_min, 1),
        'T_th': round(T_th / 1_000_000, 1), 'phi_Vc': round(phi_Vc / 1000, 1),
        'phi_Vn': round(phi_Vn / 1000, 1), 'lambda_s': round(lambda_s, 3),
        's_exact': round(s_exact, 1), 's_max': round(s_max, 1),
        'combined_stress': round(combined_stress, 2), 'stress_limit': round(stress_limit, 2)
    }


def calculate_development_length(db, fy, fc, is_top_bar, cover_clear, clear_spacing, lambda_c):
    if db == 0:
        return {'ld': 0, 'lap': 0, 'ldh': 0}
    psi_t = 1.3 if is_top_bar else 1.0
    psi_e = 1.0
    psi_s = 0.8 if db <= 20 else 1.0
    cb = min(cover_clear + db / 2, clear_spacing / 2 + db / 2)
    conf_term = min(cb / db, 2.5)
    ld_calc = (fy / (1.1 * lambda_c * math.sqrt(fc))) * ((psi_t * psi_e * psi_s) / conf_term) * db
    ld = max(ld_calc, 300)
    lap_splice = max(1.3 * ld, 300)
    ldh_calc = max((0.24 * psi_e * fy / (lambda_c * math.sqrt(fc))) * db, 8 * db, 150)
    if db > 20:
        ldh_calc *= 1.2
    return {
        'ld': math.ceil(ld / 50) * 50,
        'lap': math.ceil(lap_splice / 50) * 50,
        'ldh': math.ceil(ldh_calc / 50) * 50
    }


def create_pdf_report(b, h, fc, fy, fyt, frame_name, env_img_path, zone_data, input_mode):
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=False)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 6, "RC Beam Design Calculation Package", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 5, f"Frame ID: {frame_name}  |  Code: ACI 318-19  |  Input: {input_mode}", ln=True, align='C')
    pdf.cell(0, 5, f"Section: {b}x{h} mm  |  f'c = {fc} MPa  |  fy = {fy} MPa  |  fyt = {fyt} MPa",
             ln=True, align='C')
    pdf.ln(3)

    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 6, "1. Force Envelopes / Demands", ln=True)
    if env_img_path and os.path.exists(env_img_path):
        y_before = pdf.get_y()
        pdf.image(env_img_path, x=35, w=140)
        pdf.set_y(y_before + 105)
    else:
        pdf.set_font("Arial", 'I', 9)
        pdf.cell(0, 6, "(Manual input — no envelope diagram)", ln=True)
    pdf.ln(2)

    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 6, "2. Zone Capacities & Detailing", ln=True)
    for zone in ["Left", "Mid", "Right"]:
        data = zone_data.get(zone)
        if not data:
            continue
        pdf.set_font("Arial", 'B', 9)
        pdf.cell(15, 5, f"{zone}:", border=0)
        pdf.set_font("Arial", '', 9)
        txt1 = (f"Mu: {data['Mu']} kNm | phi*Mn: {data['phi_Mn']} kNm (D/C: {data['DC_flex']})   ||   "
                f"Vu: {data['Vu']} kN -> {data['stirrups']}")
        pdf.cell(0, 5, txt1, ln=True)
        pdf.cell(15, 5, "", border=0)
        if zone in ["Left", "Right"]:
            txt2 = f"Top Hook (ldh): {data['dev_top']} mm  |  Top Lap: {data['dev_top_lap']} mm"
        else:
            txt2 = f"Bot Lap Splice: {data['dev_bot']} mm"
        pdf.cell(0, 5, txt2, ln=True)
        pdf.ln(1)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            pdf_bytes = f.read()
    try:
        if env_img_path and os.path.exists(env_img_path):
            os.remove(env_img_path)
        os.remove(tmp_pdf.name)
    except Exception:
        pass
    return pdf_bytes


# ==========================================
# 2. THE WEB INTERFACE
# ==========================================

st.title("🏗️ RC Beam Designer — ACI 318-19")
st.caption("3-zone design: Left Support (i) · Midspan · Right Support (j)")

# ---- INPUT MODE TOGGLE ----
st.markdown("### Force Input Source")
input_mode = st.radio(
    "Select how to input beam forces:",
    ["📂 SAP2000 CSV Upload", "✏️ Manual Input"],
    horizontal=True,
    help="SAP2000 mode reads forces automatically from a CSV export. Manual mode lets you type demands directly."
)
use_sap = input_mode == "📂 SAP2000 CSV Upload"

st.markdown("---")

# ---- DEFAULT FORCES (filled either by SAP2000 or manual inputs) ----
forces = {
    'Left':  {'M': 0.0, 'V': 0.0, 'T': 0.0},
    'Mid':   {'M': 0.0, 'V': 0.0, 'T': 0.0},
    'Right': {'M': 0.0, 'V': 0.0, 'T': 0.0},
}
beam_length = 0.0
fig_env = None
env_img_path = None
selected_frame = "Manual"
df = None
max_V2_raw = 0.0
combo_V2 = "—"
stat_V2 = 0.0

# ==========================================
# PATH A — SAP2000 CSV
# ==========================================
if use_sap:
    uploaded_file = st.file_uploader("Upload RAW SAP2000 frame-forces CSV", type=["csv"])

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        if 'Frame' in df_raw.columns and str(df_raw['Frame'].iloc[0]).strip().lower() == 'text':
            df_raw = df_raw.drop(0).reset_index(drop=True)
        for col in ['Station', 'P', 'V2', 'V3', 'T', 'M2', 'M3']:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

        if 'Frame' in df_raw.columns:
            selected_frame = st.sidebar.selectbox("Select Beam (Frame)", df_raw['Frame'].unique())
            df = df_raw[df_raw['Frame'] == selected_frame].copy()
            if 'OutputCase' not in df.columns:
                df['OutputCase'] = "Manual"
            df['V2_abs'] = df['V2'].abs()
            df['T_abs']  = df['T'].abs()
            beam_length  = df['Station'].max()

            # --- Zone extraction ---
            df_left  = df[df['Station'] <= 0.1 * beam_length]
            df_right = df[df['Station'] >= 0.9 * beam_length]
            df_mid   = df[(df['Station'] > 0.3 * beam_length) & (df['Station'] < 0.7 * beam_length)]

            forces['Left']['M']  = df_left['M3'].min()  if not df_left.empty  else 0
            forces['Left']['V']  = df_left['V2_abs'].max()  if not df_left.empty  else 0
            forces['Left']['T']  = df_left['T_abs'].max()  if not df_left.empty  else 0
            forces['Right']['M'] = df_right['M3'].min() if not df_right.empty else 0
            forces['Right']['V'] = df_right['V2_abs'].max() if not df_right.empty else 0
            forces['Right']['T'] = df_right['T_abs'].max() if not df_right.empty else 0
            forces['Mid']['M']   = df['M3'].max()
            forces['Mid']['V']   = df_mid['V2_abs'].max() if not df_mid.empty else 0
            forces['Mid']['T']   = df_mid['T_abs'].max() if not df_mid.empty else 0

            # --- Info banner ---
            st.write(f"### Beam: {selected_frame}  (L = {beam_length} m)")
            h_min_req = (beam_length * 1000) / 18.5
            st.info(
                f"**Deflection control (ACI Table 9.3.1.1):** h_min ≈ {round(h_min_req, 1)} mm  "
                f"(L/18.5 — one-end-continuous). Adjust for actual end conditions.",
                icon="ℹ️"
            )

            # --- Envelope plot (Ultra Compact) ---
            df_env = df.groupby('Station').agg(
                M3_Max=('M3', 'max'), M3_Min=('M3', 'min'),
                V2_Max=('V2', 'max'), V2_Min=('V2', 'min')
            ).reset_index()

            fig_env, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 2.5), sharex=True, dpi=90)
            
            ax1.plot(df_env['Station'], df_env['M3_Max'], color='blue', linewidth=1, label='+M')
            ax1.plot(df_env['Station'], df_env['M3_Min'], color='red',  linewidth=1, label='−M')
            ax1.fill_between(df_env['Station'], df_env['M3_Min'], df_env['M3_Max'], color='gray', alpha=0.15)
            ax1.axhline(0, color='black', linewidth=0.5)
            ax1.set_ylabel("M", fontsize=7)
            ax1.invert_yaxis()
            ax1.legend(fontsize=6, loc="lower right")
            ax1.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
            ax1.set_title(f"Frame {selected_frame}", fontweight='bold', fontsize=8)
            ax1.tick_params(axis='both', which='major', labelsize=6)

            ax2.plot(df_env['Station'], df_env['V2_Max'], color='green',     linewidth=1, label='+V')
            ax2.plot(df_env['Station'], df_env['V2_Min'], color='darkorange', linewidth=1, label='−V')
            ax2.fill_between(df_env['Station'], df_env['V2_Min'], df_env['V2_Max'], color='lightgreen', alpha=0.2)
            ax2.axhline(0, color='black', linewidth=0.5)
            ax2.set_xlabel("Station (m)", fontsize=7)
            ax2.set_ylabel("V", fontsize=7)
            ax2.legend(fontsize=6, loc="upper right")
            ax2.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
            ax2.tick_params(axis='both', which='major', labelsize=6)
            
            plt.tight_layout()
            
            st.pyplot(fig_env, use_container_width=False) 

            # --- Critical demands metrics ---
            idx_pos_M = df['M3'].idxmax()
            idx_neg_M = df['M3'].idxmin()
            idx_V2    = df['V2_abs'].idxmax()
            idx_T     = df['T_abs'].idxmax()
            max_V2_raw = df.loc[idx_V2, 'V2_abs']
            combo_V2   = df.loc[idx_V2, 'OutputCase']
            stat_V2    = df.loc[idx_V2, 'Station']

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Max +M", f"{round(df.loc[idx_pos_M,'M3'],1)} kNm",
                      f"{df.loc[idx_pos_M,'OutputCase']} @ {df.loc[idx_pos_M,'Station']}m", delta_color="off")
            c2.metric("Max −M", f"{round(abs(df.loc[idx_neg_M,'M3']),1)} kNm",
                      f"{df.loc[idx_neg_M,'OutputCase']} @ {df.loc[idx_neg_M,'Station']}m", delta_color="off")
            c3.metric("Max V2", f"{round(max_V2_raw,1)} kN",
                      f"{combo_V2} @ {stat_V2}m", delta_color="off")
            c4.metric("Max T",  f"{round(df.loc[idx_T,'T_abs'],1)} kNm",
                      f"{df.loc[idx_T,'OutputCase']} @ {df.loc[idx_T,'Station']}m", delta_color="off")
        else:
            st.error("CSV must contain a 'Frame' column. Check your SAP2000 export.")
    else:
        st.info("Upload a SAP2000 CSV file to begin.")

# ==========================================
# PATH B — MANUAL INPUT
# ==========================================
else:
    st.markdown("#### Manual Force Input")
    st.caption(
        "Enter the factored design forces (Mu, Vu, Tu) for each zone directly. "
        "Moments are absolute values — sign convention is handled by zone."
    )

    beam_length = st.number_input("Beam span L (m) — used for deflection h_min note only",
                                  value=6.0, step=0.5, min_value=1.0)
    h_min_req = (beam_length * 1000) / 18.5
    st.info(
        f"**Deflection control (ACI Table 9.3.1.1):** h_min ≈ {round(h_min_req,1)} mm  "
        f"(L/18.5 — one-end-continuous). Adjust for actual end conditions.",
        icon="ℹ️"
    )

    st.markdown("---")
    man_col1, man_col2, man_col3 = st.columns(3)

    with man_col1:
        st.markdown("##### Left Support (i)")
        st.caption("Negative moment region — top bars in tension.")
        forces['Left']['M'] = st.number_input("Mu, left (kNm)", value=200.0, step=5.0,
                                               min_value=0.0, key="m_left",
                                               help="Factored bending moment magnitude at left support")
        forces['Left']['V'] = st.number_input("Vu, left (kN)",  value=150.0, step=5.0,
                                               min_value=0.0, key="v_left",
                                               help="Factored shear at distance d from left support face")
        forces['Left']['T'] = st.number_input("Tu, left (kNm)", value=0.0,   step=1.0,
                                               min_value=0.0, key="t_left",
                                               help="Factored torsion at left zone")

    with man_col2:
        st.markdown("##### Midspan")
        st.caption("Positive moment region — bottom bars in tension.")
        forces['Mid']['M'] = st.number_input("Mu, mid (kNm)", value=180.0, step=5.0,
                                              min_value=0.0, key="m_mid",
                                              help="Factored bending moment magnitude at midspan")
        forces['Mid']['V'] = st.number_input("Vu, mid (kN)",  value=60.0,  step=5.0,
                                              min_value=0.0, key="v_mid",
                                              help="Factored shear at midspan zone")
        forces['Mid']['T'] = st.number_input("Tu, mid (kNm)", value=0.0,   step=1.0,
                                              min_value=0.0, key="t_mid",
                                              help="Factored torsion at midspan")

    with man_col3:
        st.markdown("##### Right Support (j)")
        st.caption("Negative moment region — top bars in tension.")
        forces['Right']['M'] = st.number_input("Mu, right (kNm)", value=220.0, step=5.0,
                                                min_value=0.0, key="m_right",
                                                help="Factored bending moment magnitude at right support")
        forces['Right']['V'] = st.number_input("Vu, right (kN)",  value=155.0, step=5.0,
                                                min_value=0.0, key="v_right",
                                                help="Factored shear at distance d from right support face")
        forces['Right']['T'] = st.number_input("Tu, right (kNm)", value=0.0,   step=1.0,
                                                min_value=0.0, key="t_right",
                                                help="Factored torsion at right zone")

    st.markdown("---")
    st.markdown("**Demand summary (entered values):**")
    s1, s2, s3 = st.columns(3)
    for col, zone in zip([s1, s2, s3], ["Left", "Mid", "Right"]):
        col.metric(f"{zone}  Mu", f"{forces[zone]['M']} kNm")
        col.metric(f"{zone}  Vu", f"{forces[zone]['V']} kN")
        col.metric(f"{zone}  Tu", f"{forces[zone]['T']} kNm")

    selected_frame = "Manual"

# ==========================================
# SHARED — SECTION & REBAR INPUTS
# ==========================================

st.markdown("---")
col_prop, col_rebar = st.columns([1, 2])

with col_prop:
    st.header("Section & Materials")
    b = st.number_input("Width (b) mm", value=300, step=50, min_value=150)
    h = st.number_input("Total Depth (h) mm", value=600, step=50, min_value=200)
    fc = st.number_input("Concrete f'c (MPa)", value=35, step=5, min_value=20)
    lambda_c = st.selectbox("Concrete type (λ)", [1.0, 0.85, 0.75],
                             format_func=lambda x: {1.0: "1.0 — Normal weight",
                                                    0.85: "0.85 — Sand-lightweight",
                                                    0.75: "0.75 — All-lightweight"}[x])
    st.header("Transverse Steel")
    fyt = st.number_input("Stirrup fy (MPa)", value=400, step=10, min_value=240)
    bar_v_options = {'RB9': 9, 'DB10': 10, 'DB12': 12, 'DB16': 16}
    bar_v = bar_v_options[st.selectbox("Stirrup size", list(bar_v_options.keys()), index=1)]
    n_legs    = st.number_input("Stirrup legs", min_value=2, value=2, step=1)
    cover_clear = st.number_input("Clear cover to stirrup (mm)", value=40, step=5, min_value=20)
    clear_space = st.number_input("Min clear bar spacing (mm)", value=25, step=5, min_value=20)

with col_rebar:
    st.header("Zone Reinforcement")
    fy = st.number_input("Main steel fy (MPa)", value=500, step=10, min_value=300)
    bar_opts = {'DB12': 12, 'DB16': 16, 'DB20': 20, 'DB25': 25, 'DB28': 28, 'DB32': 32}

    tabs = st.tabs(["Left Support (i)", "Midspan", "Right Support (j)"])
    rebar_data    = {}
    bar_selections = {}

    for i, zone in enumerate(["Left", "Mid", "Right"]):
        with tabs[i]:
            def_t = 4 if zone in ["Left", "Right"] else 2
            def_b = 4 if zone == "Mid" else 2

            c1z, c2z = st.columns(2)
            with c1z:
                st.write("**Top Bars**")
                t1c, t1s = st.columns(2)
                t_n1      = t1c.number_input("L1 n", 0, value=def_t, key=f"t1_{zone}")
                t_d1_name = t1s.selectbox("L1 size", list(bar_opts.keys()), index=3, key=f"td1_{zone}")
                t2c, t2s  = st.columns(2)
                t_n2      = t2c.number_input("L2 n", 0, value=0, key=f"t2_{zone}")
                t_d2_name = t2s.selectbox("L2 size", list(bar_opts.keys()), index=2, key=f"td2_{zone}")
                t3c, t3s  = st.columns(2)
                t_n3      = t3c.number_input("L3 n", 0, value=0, key=f"t3_{zone}")
                t_d3_name = t3s.selectbox("L3 size", list(bar_opts.keys()), index=2, key=f"td3_{zone}")

            with c2z:
                st.write("**Bottom Bars**")
                b1c, b1s = st.columns(2)
                b_n1      = b1c.number_input("L1 n", 0, value=def_b, key=f"b1_{zone}")
                b_d1_name = b1s.selectbox("L1 size", list(bar_opts.keys()), index=3, key=f"bd1_{zone}")
                b2c, b2s  = st.columns(2)
                b_n2      = b2c.number_input("L2 n", 0, value=0, key=f"b2_{zone}")
                b_d2_name = b2s.selectbox("L2 size", list(bar_opts.keys()), index=2, key=f"bd2_{zone}")
                b3c, b3s  = st.columns(2)
                b_n3      = b3c.number_input("L3 n", 0, value=0, key=f"b3_{zone}")
                b_d3_name = b3s.selectbox("L3 size", list(bar_opts.keys()), index=2, key=f"bd3_{zone}")

            t_d1, t_d2, t_d3 = bar_opts[t_d1_name], bar_opts[t_d2_name], bar_opts[t_d3_name]
            b_d1, b_d2, b_d3 = bar_opts[b_d1_name], bar_opts[b_d2_name], bar_opts[b_d3_name]

            rebar_data[zone] = {
                'top': get_rebar_group(t_n1, t_d1, t_n2, t_d2, t_n3, t_d3, cover_clear, bar_v, clear_space),
                'bot': get_rebar_group(b_n1, b_d1, b_n2, b_d2, b_n3, b_d3, cover_clear, bar_v, clear_space),
            }
            bar_selections[zone] = {
                'top_d1': t_d1 if t_n1 > 0 else 0,
                'bot_d1': b_d1 if b_n1 > 0 else 0,
            }

# ==========================================
# RUN DESIGN
# ==========================================

st.markdown("---")
if st.button("🚀 Run Full 3-Zone Detailing Design", type="primary", use_container_width=True):

    # Save envelope figure for PDF
    if fig_env is not None:
        tmp_env = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig_env.savefig(tmp_env.name, bbox_inches='tight', dpi=150)
        env_img_path = tmp_env.name
    else:
        env_img_path = None

    cols = st.columns(3)
    pdf_zone_data = {}

    for idx, zone in enumerate(["Left", "Mid", "Right"]):
        with cols[idx]:
            st.subheader(f"{zone} Section")
            top_rg = rebar_data[zone]['top']
            bot_rg = rebar_data[zone]['bot']

            # Bar fit check
            if top_rg.width_req > b or bot_rg.width_req > b:
                st.error(f"🚨 Bars don't fit in {b} mm width. "
                         f"Top needs {round(top_rg.width_req,1)} mm, "
                         f"Bot needs {round(bot_rg.width_req,1)} mm.")
                pdf_zone_data[zone] = None
                continue

            # Effective depths
            if zone == "Mid":
                As_tens, As_comp = bot_rg.area, top_rg.area
                d       = h - bot_rg.centroid
                dt      = h - bot_rg.extreme_fiber
                d_prime = top_rg.centroid
            else:
                As_tens, As_comp = top_rg.area, bot_rg.area
                d       = h - top_rg.centroid
                dt      = h - top_rg.extreme_fiber
                d_prime = bot_rg.centroid

            Mu = abs(forces[zone]['M'])

            # ---- FLEXURE ----
            res_flex = calculate_beam_flexure(b, h, d, dt, d_prime, fc, fy, As_tens, As_comp)
            DC_flex  = round(Mu / res_flex['phi_Mn'], 2) if res_flex['phi_Mn'] > 0 else 999.9

            st.write(f"**Flexure:** φMn = {res_flex['phi_Mn']} kNm "
                     f"(Req: {round(Mu,1)}) | **D/C: {DC_flex}**")
            if not res_flex['converged']:
                st.error("🚨 Solver convergence failed.")
            elif not res_flex['passes_As_min']:
                st.warning(f"⚠️ As < As,min ({res_flex['As_min']} mm²)")
            elif not res_flex['is_ductile']:
                st.error("❌ Over-reinforced — compression-controlled.")
            elif res_flex['phi_Mn'] >= Mu:
                st.success("✅ Flexure OK")
            else:
                st.error("❌ Flexure fails")

            # ---- SHEAR ----
            if use_sap and df is not None and beam_length > 0 and zone in ["Left", "Right"]:
                d_m = d / 1000.0
                valid = df[(df['Station'] >= d_m) & (df['Station'] <= (beam_length - d_m))]
                if not valid.empty:
                    if zone == "Left":
                        sub = valid[valid['Station'] <= 0.3 * beam_length]
                    else:
                        sub = valid[valid['Station'] >= 0.7 * beam_length]
                    idx_v = sub['V2_abs'].idxmax() if not sub.empty else valid['V2_abs'].idxmax()
                    Vu_design   = valid.loc[idx_v, 'V2_abs']
                    combo_label = f"{valid.loc[idx_v,'OutputCase']} @ {round(valid.loc[idx_v,'Station'],2)}m"
                else:
                    Vu_design   = max_V2_raw
                    combo_label = combo_V2
            else:
                Vu_design   = forces[zone]['V']
                combo_label = "Manual input"

            res_shear = calculate_shear_torsion(
                b, h, d, fc, fyt, fy, cover_clear,
                Vu_design, forces[zone]['T'],
                n_legs, bar_v, lambda_c
            )
            phi_Vn  = res_shear['phi_Vn']
            DC_shear = round(Vu_design / phi_Vn, 2) if phi_Vn > 0 else 999.9

            st.write(f"**Shear:** Vu = {round(Vu_design,1)} kN ({combo_label})")
            if res_shear['final_s'] > 0:
                st.success(
                    f"✅ {n_legs}-DB{bar_v} @ {res_shear['final_s']} mm  "
                    f"| φVn = {phi_Vn} kN (D/C: {DC_shear})"
                )
                if res_shear['needs_torsion']:
                    st.info(f"Torsion governs — add Al = {res_shear['Al_req']} mm² (min {res_shear['Al_min']} mm²)")
            elif res_shear['section_fails']:
                st.error("❌ Section too small — web crushing limit exceeded.")
            else:
                st.error("❌ Spacing too tight (< 50 mm). Increase stirrup size or legs.")

            # ---- DEVELOPMENT LENGTHS ----
            dev_top = calculate_development_length(
                bar_selections[zone]['top_d1'], fy, fc, True, cover_clear, clear_space, lambda_c)
            dev_bot = calculate_development_length(
                bar_selections[zone]['bot_d1'], fy, fc, False, cover_clear, clear_space, lambda_c)

            st.write("**Development lengths:**")
            if zone in ["Left", "Right"]:
                st.write(f"- Top hook (ldh): **{dev_top['ldh']}** mm")
                st.write(f"- Top lap splice: **{dev_top['lap']}** mm")
            else:
                st.write(f"- Bottom lap splice: **{dev_bot['lap']}** mm")

            # ---- EXPANDER: full calc steps ----
            with st.expander("🧮 Calculation steps"):
                st.markdown(f"""
**1. Flexural strain compatibility**
- d = {round(d,1)} mm, dt = {round(dt,1)} mm, d' = {round(d_prime,1)} mm
- Neutral axis c = **{res_flex['c']}** mm
- εt = **{res_flex['eps_t']}** → φ = {res_flex['phi']}
- Mn = **{res_flex['Mn']}** kNm → φMn = **{res_flex['phi_Mn']}** kNm

**2. Shear & torsion**
- λs = {res_shear['lambda_s']} (size effect, ACI §22.5.5.1.3)
- φVc = **{res_shear['phi_Vc']}** kN
- φVn = **{res_shear['phi_Vn']}** kN
- s_req = {res_shear['s_exact']} mm → s_max = {res_shear['s_max']} mm → **use {res_shear['final_s']} mm**
- Combined stress = {res_shear['combined_stress']} MPa  (limit {res_shear['stress_limit']} MPa)
                """)

            pdf_zone_data[zone] = {
                'Mu': round(Mu, 1), 'phi_Mn': res_flex['phi_Mn'], 'DC_flex': DC_flex,
                'Vu': round(Vu_design, 1), 'DC_shear': DC_shear,
                'stirrups': (f"{n_legs}-DB{bar_v} @ {res_shear['final_s']} mm (D/C: {DC_shear})"
                             if res_shear['final_s'] > 0 else "FAILS"),
                'dev_top': dev_top['ldh'], 'dev_top_lap': dev_top['lap'], 'dev_bot': dev_bot['lap']
            }

    # ---- PDF EXPORT ----
    if any(v is not None for v in pdf_zone_data.values()):
        st.markdown("---")
        input_label = "SAP2000 CSV" if use_sap else "Manual Input"
        pdf_bytes = create_pdf_report(
            b, h, fc, fy, fyt, selected_frame,
            env_img_path, pdf_zone_data, input_label
        )
        st.download_button(
            label="📄 Download PDF Calculation Report",
            data=pdf_bytes,
            file_name=f"Beam_{selected_frame}_Report.pdf",
            mime="application/pdf",
            type="primary"
        )
