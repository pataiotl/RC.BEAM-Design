import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    if As == 0: return {'phi_Mn': 0, 'passes_As_min': False, 'is_ductile': False, 'converged': True, 'As_min': 0,
                        'c': 0, 'a': 0, 'eps_t': 0, 'phi': 0, 'Mn': 0}
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
        if d_prime <= a and eps_s_prime > 0: Cs -= As_prime * 0.85 * fc
        eps_s = ecu * (d - c) / c if c > 0 else 0
        fs = min(fy, max(-fy, eps_s * Es))
        T = As * fs
        if (Cc + Cs) <= T:
            converged = True
            break
        c -= 0.1

    Mn_Nmm = Cc * (d - a / 2) + Cs * (d - d_prime)
    Mn_kNm = Mn_Nmm / 1000000
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
    if d <= 0: return {'final_s': 0, 'section_fails': True, 'needs_torsion': False, 'Al_req': 0, 'Al_min': 0,
                       's_exact': 0, 's_max': 0, 'combined_stress': 0, 'stress_limit': 0, 'lambda_s': 1.0, 'phi_Vc': 0,
                       'phi_Vn': 0, 'T_th': 0}
    phi_v = 0.75
    Vu, Tu = abs(Vu_kN) * 1000, abs(Tu_kNm) * 1000000
    A_leg = math.pi * (bar_dia ** 2) / 4
    lambda_s = min(1.0, math.sqrt(2 / (1 + 0.004 * d)))

    Vc = 0.17 * lambda_s * lambda_c * math.sqrt(fc) * b * d
    phi_Vc = phi_v * Vc

    x1, y1 = b - 2 * (cover_clear + bar_dia / 2), h - 2 * (cover_clear + bar_dia / 2)
    Aoh, Ao, ph, Acp, pcp = x1 * y1, 0.85 * (x1 * y1), 2 * (x1 + y1), b * h, 2 * (b + h)

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
        'final_s': 0 if final_s < 50 or section_fails else final_s, 'section_fails': section_fails,
        'needs_torsion': needs_torsion, 'Al_req': round(Al_req, 1), 'Al_min': round(Al_min, 1),
        'T_th': round(T_th / 1000000, 1), 'phi_Vc': round(phi_Vc / 1000, 1), 'phi_Vn': round(phi_Vn / 1000, 1),
        'lambda_s': round(lambda_s, 3), 's_exact': round(s_exact, 1), 's_max': round(s_max, 1),
        'combined_stress': round(combined_stress, 2), 'stress_limit': round(stress_limit, 2)
    }


def calculate_development_length(db, fy, fc, is_top_bar, cover_clear, clear_spacing, lambda_c):
    if db == 0: return {'ld': 0, 'lap': 0, 'ldh': 0}
    psi_t = 1.3 if is_top_bar else 1.0
    psi_e = 1.0
    psi_s = 0.8 if db <= 20 else 1.0
    cb = min(cover_clear + db / 2, clear_spacing / 2 + db / 2)
    conf_term = min(cb / db, 2.5)

    ld_calc = (fy / (1.1 * lambda_c * math.sqrt(fc))) * ((psi_t * psi_e * psi_s) / conf_term) * db
    ld = max(ld_calc, 300)
    lap_splice = max(1.3 * ld, 300)
    ldh_calc = max((0.24 * psi_e * fy / (lambda_c * math.sqrt(fc))) * db, 8 * db, 150)
    if db > 20: ldh_calc *= 1.2

    return {
        'ld': math.ceil(ld / 50) * 50,
        'lap': math.ceil(lap_splice / 50) * 50,
        'ldh': math.ceil(ldh_calc / 50) * 50
    }


def draw_cross_section(b, h, cover_clear, bar_v, top_rg: RebarGroup, bot_rg: RebarGroup, title, stirrup_label=""):
    """Draws the cross section with dynamic leader line callouts for rebars and stirrups."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)

    ax.add_patch(patches.Rectangle((0, 0), b, h, linewidth=1.5, edgecolor='black', facecolor='lightgray', alpha=0.3))
    ax.add_patch(patches.Rectangle((cover_clear, cover_clear), b - 2 * cover_clear, h - 2 * cover_clear, linewidth=1.5,
                                   edgecolor='blue', facecolor='none'))

    # Add dynamic Stirrup Callout pointing to the left leg
    if stirrup_label:
        ax.annotate(
            stirrup_label,
            xy=(cover_clear, h / 2),
            xytext=(-20, h / 2),
            fontsize=8, fontweight='bold', color='blue',
            ha='right', va='center',
            arrowprops=dict(arrowstyle="->", color='blue', lw=1.2)
        )

    def plot_bars_with_callouts(rg, is_top, color):
        for idx, layer in enumerate(rg.layers):
            n_bars, dia, y_dist = layer
            if n_bars == 0: continue

            y_actual = (h - y_dist) if is_top else y_dist
            rightmost_x = b / 2

            if n_bars == 1:
                ax.add_patch(patches.Circle((b / 2, y_actual), dia / 2, facecolor=color, edgecolor='black', zorder=3))
                rightmost_x = b / 2 + dia / 2
            elif n_bars > 1:
                x_start = cover_clear + bar_v + dia / 2
                spacing = (b - 2 * (cover_clear + bar_v) - dia) / (n_bars - 1)
                for i in range(n_bars):
                    bar_x = x_start + i * spacing
                    ax.add_patch(
                        patches.Circle((bar_x, y_actual), dia / 2, facecolor=color, edgecolor='black', zorder=3))
                    if i == n_bars - 1: rightmost_x = bar_x + dia / 2

            label = f"{int(n_bars)}-DB{int(dia)}"
            y_offset = (20 + idx * 15) if is_top else (-20 - idx * 15)

            ax.annotate(
                label,
                xy=(rightmost_x, y_actual),
                xytext=(b + 25, y_actual + y_offset),
                fontsize=8, fontweight='bold', color=color,
                ha='left', va='center',
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2)
            )

    plot_bars_with_callouts(top_rg, True, 'crimson')
    plot_bars_with_callouts(bot_rg, False, 'mediumseagreen')

    ax.text(b / 2, -25, f"b = {int(b)}", ha='center', fontsize=8)
    ax.text(-25, h / 2, f"h = {int(h)}", va='center', rotation=90, fontsize=8)

    ax.set_xlim(-100, b + 120)
    ax.set_ylim(-50, h + 50)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontweight="bold", fontsize=10)
    return fig


def create_pdf_report(b, h, fc, fy, fyt, frame_name, env_img_path, zone_data):
    """Generates a 1-Page A4 PDF with Cross Sections Side-by-Side at the bottom."""
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=False)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 6, "RC Beam Design Calculation Package", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 5, f"Frame ID: {frame_name}  |  Code: ACI 318-19  |  Section: {b}x{h} mm", ln=True, align='C')
    pdf.cell(0, 5, f"Materials: f'c = {fc} MPa  |  fy = {fy} MPa  |  fyt = {fyt} MPa", ln=True, align='C')
    pdf.ln(3)

    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 6, "1. Force Envelopes", ln=True)
    y_before_img = pdf.get_y()
    pdf.image(env_img_path, x=35, w=140)
    pdf.set_y(y_before_img + 105)
    pdf.ln(2)

    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 6, "2. Zone Capacities & Detailing", ln=True)

    for zone in ["Left", "Mid", "Right"]:
        data = zone_data[zone]
        pdf.set_font("Arial", 'B', 9)
        pdf.cell(15, 5, f"{zone}:", border=0)
        pdf.set_font("Arial", '', 9)

        if zone in ["Left", "Right"]:
            txt1 = f"Mu: {data['Mu']} kNm | phi*Mn: {data['phi_Mn']} kNm (D/C: {data['DC_flex']})   ||   Vu: {data['Vu']} kN -> {data['stirrups']}"
            pdf.cell(0, 5, txt1, ln=True)
            pdf.cell(15, 5, "", border=0)
            txt2 = f"Top Hook (ldh): {data['dev_top']} mm  |  Top Lap Splice: {data['dev_top_lap']} mm"
            pdf.cell(0, 5, txt2, ln=True)
        else:
            txt1 = f"Mu: {data['Mu']} kNm | phi*Mn: {data['phi_Mn']} kNm (D/C: {data['DC_flex']})   ||   Vu: {data['Vu']} kN -> {data['stirrups']}"
            pdf.cell(0, 5, txt1, ln=True)
            pdf.cell(15, 5, "", border=0)
            txt2 = f"Bot Lap Splice: {data['dev_bot']} mm"
            pdf.cell(0, 5, txt2, ln=True)
        pdf.ln(1)

    pdf.ln(2)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 6, "3. Cross-Section Detailing", ln=True)

    y_cs = pdf.get_y()
    x_positions = {'Left': 15, 'Mid': 75, 'Right': 135}

    for zone in ["Left", "Mid", "Right"]:
        pdf.image(zone_data[zone]['img_path'], x=x_positions[zone], y=y_cs, w=60)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            pdf_bytes = f.read()

    try:
        os.remove(env_img_path)
        for d in zone_data.values(): os.remove(d['img_path'])
        os.remove(tmp_pdf.name)
    except:
        pass

    return pdf_bytes


# ==========================================
# 2. THE WEB INTERFACE
# ==========================================

st.title("🏗️ RC Beam Designer (Detailed Sections)")
st.write("Extracts forces and details 3 critical zones: Left Support (i), Midspan, and Right Support (j).")

uploaded_file = st.file_uploader("Upload RAW SAP2000 loads (CSV)", type=["csv"])

beam_length = 0
forces = {'Left': {'M': 0, 'V': 0, 'T': 0}, 'Mid': {'M': 0, 'V': 0, 'T': 0}, 'Right': {'M': 0, 'V': 0, 'T': 0}}
max_V2_raw = 0
combo_V2 = "Manual"
stat_V2 = 0
fig_env = None

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    if 'Frame' in df_raw.columns and str(df_raw['Frame'].iloc[0]).strip().lower() == 'text': df_raw = df_raw.drop(
        0).reset_index(drop=True)
    for col in ['Station', 'P', 'V2', 'V3', 'T', 'M2', 'M3']:
        if col in df_raw.columns: df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    if 'Frame' in df_raw.columns:
        selected_frame = st.sidebar.selectbox("Select Beam (Frame)", df_raw['Frame'].unique())
        df = df_raw[df_raw['Frame'] == selected_frame].copy()
        df['V2_abs'], df['T_abs'] = df['V2'].abs(), df['T'].abs()
        beam_length = df['Station'].max()

        df_left = df[df['Station'] <= 0.1 * beam_length]
        forces['Left']['M'] = df_left['M3'].min() if not df_left.empty else 0
        forces['Left']['V'] = df_left['V2_abs'].max() if not df_left.empty else 0
        forces['Left']['T'] = df_left['T_abs'].max() if not df_left.empty else 0

        df_right = df[df['Station'] >= 0.9 * beam_length]
        forces['Right']['M'] = df_right['M3'].min() if not df_right.empty else 0
        forces['Right']['V'] = df_right['V2_abs'].max() if not df_right.empty else 0
        forces['Right']['T'] = df_right['T_abs'].max() if not df_right.empty else 0

        # Extract precise midspan shear for stirrup callouts
        df_mid = df[(df['Station'] > 0.3 * beam_length) & (df['Station'] < 0.7 * beam_length)]
        forces['Mid']['M'] = df['M3'].max()
        forces['Mid']['V'] = df_mid['V2_abs'].max() if not df_mid.empty else 0
        forces['Mid']['T'] = df_mid['T_abs'].max() if not df_mid.empty else 0

        st.write(f"### Designing Beam: {selected_frame} (L = {beam_length} m)")
        h_min_req = (beam_length * 1000) / 18.5
        st.info(f"**Deflection Control:** ACI $h_{{min}} \\approx {round(h_min_req, 1)}$ mm (L/18.5)")

        df_env = df.groupby('Station').agg(M3_Max=('M3', 'max'), M3_Min=('M3', 'min'), V2_Max=('V2', 'max'),
                                           V2_Min=('V2', 'min')).reset_index()
        fig_env, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 3), sharex=True, dpi=120)

        ax1.plot(df_env['Station'], df_env['M3_Max'], color='blue', linewidth=1)
        ax1.plot(df_env['Station'], df_env['M3_Min'], color='red', linewidth=1)
        ax1.fill_between(df_env['Station'], df_env['M3_Min'], df_env['M3_Max'], color='gray', alpha=0.2)
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.set_title(f"Force Envelopes - Frame {selected_frame}", fontweight="bold", fontsize=8)
        ax1.set_ylabel("M (kNm)", fontsize=7)
        ax1.invert_yaxis()
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.tick_params(axis='both', labelsize=6)

        ax2.plot(df_env['Station'], df_env['V2_Max'], color='green', linewidth=1)
        ax2.plot(df_env['Station'], df_env['V2_Min'], color='darkorange', linewidth=1)
        ax2.fill_between(df_env['Station'], df_env['V2_Min'], df_env['V2_Max'], color='lightgreen', alpha=0.2)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_xlabel("Station (m)", fontsize=7)
        ax2.set_ylabel("V (kN)", fontsize=7)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.tick_params(axis='both', labelsize=6)

        plt.tight_layout()
        st.pyplot(fig_env, use_container_width=False)

        if 'OutputCase' not in df.columns: df['OutputCase'] = "Manual"
        idx_pos_M = df['M3'].idxmax()
        max_pos_M = df.loc[idx_pos_M, 'M3']
        combo_pos_M = df.loc[idx_pos_M, 'OutputCase']
        stat_pos_M = df.loc[idx_pos_M, 'Station']

        idx_neg_M = df['M3'].idxmin()
        max_neg_M = df.loc[idx_neg_M, 'M3']
        combo_neg_M = df.loc[idx_neg_M, 'OutputCase']
        stat_neg_M = df.loc[idx_neg_M, 'Station']

        idx_V2 = df['V2_abs'].idxmax()
        max_V2_raw = df.loc[idx_V2, 'V2_abs']
        combo_V2 = df.loc[idx_V2, 'OutputCase']
        stat_V2 = df.loc[idx_V2, 'Station']

        idx_T = df['T_abs'].idxmax()
        max_T = df.loc[idx_T, 'T_abs']
        combo_T = df.loc[idx_T, 'OutputCase']
        stat_T = df.loc[idx_T, 'Station']

        st.markdown("### Critical Demands & Governing Combos:")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max +M", f"{round(max_pos_M, 1)} kNm", f"Combo: {combo_pos_M} @ {stat_pos_M}m", delta_color="off")
        c2.metric("Max -M", f"{round(abs(max_neg_M), 1)} kNm", f"Combo: {combo_neg_M} @ {stat_neg_M}m",
                  delta_color="off")
        c3.metric("Max Shear (V2)", f"{round(max_V2_raw, 1)} kN", f"Combo: {combo_V2} @ {stat_V2}m", delta_color="off")
        c4.metric("Max Torsion (T)", f"{round(max_T, 1)} kNm", f"Combo: {combo_T} @ {stat_T}m", delta_color="off")

st.markdown("---")

col_prop, col_rebar = st.columns([1, 2])

with col_prop:
    st.header("1. Section & Materials")
    b = st.number_input("Width (b) mm", value=300, step=50)
    h = st.number_input("Total Depth (h) mm", value=600, step=50)
    fc = st.number_input("Concrete f'c (MPa)", value=35, step=5)
    lambda_c = st.selectbox("Concrete (λ)", [1.0, 0.85, 0.75])

    st.header("2. Transverse Steel")
    fyt = st.number_input("Stirrup fy (MPa)", value=400, step=10)
    bar_v_options = {'RB9': 9, 'DB10': 10, 'DB12': 12, 'DB16': 16}
    bar_v = bar_v_options[st.selectbox("Stirrup Size", list(bar_v_options.keys()), index=1)]
    n_legs = st.number_input("Stirrup Legs", min_value=2, value=2, step=1)
    cover_clear = st.number_input("Clear Cover (mm)", value=40, step=5)
    clear_space = st.number_input("Clear Spacing (mm)", value=25, step=5)

with col_rebar:
    st.header("3. Zone Reinforcement")
    fy = st.number_input("Main Steel fy (MPa)", value=500, step=10)
    bar_opts = {'DB12': 12, 'DB16': 16, 'DB20': 20, 'DB25': 25, 'DB28': 28, 'DB32': 32}

    tabs = st.tabs(["Left Support (i)", "Midspan", "Right Support (j)"])
    rebar_data = {}
    bar_selections = {}

    for i, zone in enumerate(["Left", "Mid", "Right"]):
        with tabs[i]:
            st.write(f"**{zone} Zone Reinforcement**")
            def_t = 4 if zone in ["Left", "Right"] else 2
            def_b = 4 if zone == "Mid" else 2

            c1, c2 = st.columns(2)
            with c1:
                st.write("**Top Bars**")
                t1_c, t1_s = st.columns(2)
                t_n1 = t1_c.number_input(f"L1 Bars", 0, value=def_t, key=f"t1_{zone}")
                t_d1_name = t1_s.selectbox(f"L1 Size", list(bar_opts.keys()), index=3, key=f"td1_{zone}")

                t2_c, t2_s = st.columns(2)
                t_n2 = t2_c.number_input(f"L2 Bars", 0, value=0, key=f"t2_{zone}")
                t_d2_name = t2_s.selectbox(f"L2 Size", list(bar_opts.keys()), index=2, key=f"td2_{zone}")

                t3_c, t3_s = st.columns(2)
                t_n3 = t3_c.number_input(f"L3 Bars", 0, value=0, key=f"t3_{zone}")
                t_d3_name = t3_s.selectbox(f"L3 Size", list(bar_opts.keys()), index=2, key=f"td3_{zone}")

            with c2:
                st.write("**Bottom Bars**")
                b1_c, b1_s = st.columns(2)
                b_n1 = b1_c.number_input(f"L1 Bars", 0, value=def_b, key=f"b1_{zone}")
                b_d1_name = b1_s.selectbox(f"L1 Size", list(bar_opts.keys()), index=3, key=f"bd1_{zone}")

                b2_c, b2_s = st.columns(2)
                b_n2 = b2_c.number_input(f"L2 Bars", 0, value=0, key=f"b2_{zone}")
                b_d2_name = b2_s.selectbox(f"L2 Size", list(bar_opts.keys()), index=2, key=f"bd2_{zone}")

                b3_c, b3_s = st.columns(2)
                b_n3 = b3_c.number_input(f"L3 Bars", 0, value=0, key=f"b3_{zone}")
                b_d3_name = b3_s.selectbox(f"L3 Size", list(bar_opts.keys()), index=2, key=f"bd3_{zone}")

            t_d1, t_d2, t_d3 = bar_opts[t_d1_name], bar_opts[t_d2_name], bar_opts[t_d3_name]
            b_d1, b_d2, b_d3 = bar_opts[b_d1_name], bar_opts[b_d2_name], bar_opts[b_d3_name]

            rebar_data[zone] = {
                'top': get_rebar_group(t_n1, t_d1, t_n2, t_d2, t_n3, t_d3, cover_clear, bar_v, clear_space),
                'bot': get_rebar_group(b_n1, b_d1, b_n2, b_d2, b_n3, b_d3, cover_clear, bar_v, clear_space)
            }
            bar_selections[zone] = {
                'top_d1': t_d1 if t_n1 > 0 else 0,
                'bot_d1': b_d1 if b_n1 > 0 else 0
            }

st.markdown("---")

if st.button("🚀 Run Full 3-Zone Detailing Design", type="primary", use_container_width=True):
    env_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    if fig_env: fig_env.savefig(env_img.name, bbox_inches='tight', dpi=150)

    cols = st.columns(3)
    zones = ["Left", "Mid", "Right"]
    pdf_zone_data = {}

    for idx, zone in enumerate(zones):
        with cols[idx]:
            st.subheader(f"{zone} Section")
            top_rg, bot_rg = rebar_data[zone]['top'], rebar_data[zone]['bot']

            if top_rg.width_req > b or bot_rg.width_req > b:
                st.error("🚨 Bars do not physically fit.")
                continue

            Mu = abs(forces[zone]['M'])
            d = h - bot_rg.centroid if zone == "Mid" else h - top_rg.centroid
            dt = h - bot_rg.extreme_fiber if zone == "Mid" else h - top_rg.extreme_fiber
            d_prime = top_rg.centroid if zone == "Mid" else bot_rg.centroid
            As_tens = bot_rg.area if zone == "Mid" else top_rg.area
            As_comp = top_rg.area if zone == "Mid" else bot_rg.area

            res_flex = calculate_beam_flexure(b, h, d, dt, d_prime, fc, fy, As_tens, As_comp)
            DC_flex = round(Mu / res_flex['phi_Mn'], 2) if res_flex['phi_Mn'] > 0 else 999.9

            st.write(f"**Flexure:** $\phi M_n$ = {res_flex['phi_Mn']} kNm (Req: {round(Mu, 1)}) | **D/C: {DC_flex}**")
            if res_flex['phi_Mn'] >= Mu and res_flex['passes_As_min'] and res_flex['is_ductile']:
                st.success("✅ Flexure OK")
            else:
                st.error("❌ Flexure Fails / Limits")

            # Process Shear For All Zones
            if 'df' in locals() and beam_length > 0 and zone in ["Left", "Right"]:
                d_shear_m = d / 1000.0
                valid_stations = df[(df['Station'] >= d_shear_m) & (df['Station'] <= (beam_length - d_shear_m))]
                if not valid_stations.empty:
                    if zone == "Left":
                        idx_V2_d = valid_stations[valid_stations['Station'] <= 0.3 * beam_length]['V2_abs'].idxmax()
                    else:
                        idx_V2_d = valid_stations[valid_stations['Station'] >= 0.7 * beam_length]['V2_abs'].idxmax()
                    Vu_at_d = valid_stations.loc[idx_V2_d, 'V2_abs']
                    combo_V2_d = valid_stations.loc[idx_V2_d, 'OutputCase']
                    stat_V2_d = valid_stations.loc[idx_V2_d, 'Station']
                else:
                    Vu_at_d, combo_V2_d, stat_V2_d = max_V2_raw, combo_V2, stat_V2
            else:
                Vu_at_d = forces[zone]['V']
                combo_V2_d = "Midspan"
                stat_V2_d = beam_length / 2

            res_shear = calculate_shear_torsion(b, h, d, fc, fyt, fy, cover_clear, Vu_at_d, forces[zone]['T'], n_legs,
                                                bar_v, lambda_c)
            phi_Vn = res_shear.get('phi_Vn', 0)
            DC_shear = round(Vu_at_d / phi_Vn, 2) if phi_Vn > 0 else 999.9

            if zone in ["Left", "Right"]:
                st.write(f"**Shear:** $V_u$ = {round(Vu_at_d, 1)} kN (Combo: `{combo_V2_d}` @ {round(stat_V2_d, 2)}m)")
            else:
                st.write(f"**Shear:** nominal check for midspan ($V_u$ = {round(Vu_at_d, 1)} kN)")

            if res_shear['final_s'] > 0:
                st.success(
                    f"✅ Use {n_legs}-DB{bar_v} @ {res_shear['final_s']} mm | **$\phi V_n$: {phi_Vn} kN (D/C: {DC_shear})**")
                if res_shear['needs_torsion']: st.info(f"Torsion Governs. Add $A_l$ = {res_shear['Al_req']} mm²")
            else:
                if res_shear['section_fails']:
                    st.error("❌ Shear section fails ACI crushing limits.")
                else:
                    st.error("❌ Spacing too tight.")

            st.write("**Development Lengths:**")
            dev_top = calculate_development_length(bar_selections[zone]['top_d1'], fy, fc, True, cover_clear,
                                                   clear_space, lambda_c)
            dev_bot = calculate_development_length(bar_selections[zone]['bot_d1'], fy, fc, False, cover_clear,
                                                   clear_space, lambda_c)

            if zone in ["Left", "Right"]:
                st.write(f"- Top Hook ($l_{{dh}}$): **{dev_top['ldh']}** mm")
                st.write(f"- Top Splice: **{dev_top['lap']}** mm")
            else:
                st.write(f"- Bot Splice: **{dev_bot['lap']}** mm")

            # --- RENDER CALLOUT CROSS SECTION ---
            stirrup_text = f"{n_legs}-DB{bar_v} @ {res_shear.get('final_s', 0)}" if res_shear.get('final_s',
                                                                                                  0) > 0 else "Stirrup FAILS"
            fig_cs = draw_cross_section(b, h, cover_clear, bar_v, top_rg, bot_rg, f"{zone}", stirrup_label=stirrup_text)
            st.pyplot(fig_cs, use_container_width=False)
            cs_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig_cs.savefig(cs_img.name, bbox_inches='tight', dpi=150)
            plt.close(fig_cs)

            # Package data for PDF
            pdf_zone_data[zone] = {
                'Mu': round(Mu, 1), 'phi_Mn': res_flex['phi_Mn'], 'DC_flex': DC_flex,
                'Vu': round(Vu_at_d, 1), 'DC_shear': DC_shear,
                'stirrups': f"{n_legs}-DB{bar_v} @ {res_shear.get('final_s', 0)} mm (D/C: {DC_shear})" if res_shear.get(
                    'final_s', 0) > 0 else "FAILS",
                'dev_top': dev_top['ldh'], 'dev_top_lap': dev_top['lap'], 'dev_bot': dev_bot['lap'],
                'img_path': cs_img.name
            }

            with st.expander("🧮 View Calculation Steps"):
                st.markdown(f"""
                **1. Flexural Strain Compatibility**
                * $d = {round(d, 1)}$ mm, $d_t = {round(dt, 1)}$ mm, $d' = {round(d_prime, 1)}$ mm
                * Neutral Axis ($c$): **{res_flex['c']}** mm
                * Extreme Strain ($\epsilon_t$): **{res_flex['eps_t']}** $\\rightarrow$ $\phi = {res_flex['phi']}$
                * Nominal Moment ($M_n$): **{res_flex['Mn']}** kNm

                **2. Shear & Torsion Capacities**
                * Concrete Shear ($\phi V_c$): **{res_shear.get('phi_Vc', '-')}** kN
                * Total Shear ($\phi V_n$): **{res_shear.get('phi_Vn', '-')}** kN
                * Exact Spacing Req ($s_{{req}}$): **{res_shear.get('s_exact', '-')}** mm
                * Web Crushing Stress: **{res_shear.get('combined_stress', '-')}** MPa vs Limit: **{res_shear.get('stress_limit', '-')}** MPa
                """)

    # --- GENERATE PDF REPORT ---
    frame_name = selected_frame if 'selected_frame' in locals() else "Manual"
    pdf_bytes = create_pdf_report(b, h, fc, fy, fyt, frame_name, env_img.name, pdf_zone_data)

    st.markdown("---")
    st.success("✅ Design checks passed and calculation package compiled.")

    st.download_button(
        label="📄 Download 1-Page PDF Calculation Report",
        data=pdf_bytes,
        file_name=f"Beam_{frame_name}_Report.pdf",
        mime="application/pdf",
        type="primary"
    )