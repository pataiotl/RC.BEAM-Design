import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import math
import tempfile
import os
from dataclasses import dataclass
from fpdf import FPDF

# ============================================================
# PAGE CONFIG & CSS
# ============================================================
st.set_page_config(page_title="RC Beam Designer", page_icon="🏗️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
html, body, [class*="css"] { font-size: 13px; }
h1 { font-size: 1.35rem !important; font-weight: 700 !important; letter-spacing: -0.5px; }
h2 { font-size: 1.05rem !important; font-weight: 600 !important; }
h3 { font-size: 0.95rem !important; font-weight: 600 !important; }
[data-testid="stSidebar"] { min-width: 270px !important; max-width: 300px !important; }
[data-testid="stSidebar"] .block-container { padding: 0.75rem 0.75rem 1rem !important; }
[data-testid="stSidebar"] label { font-size: 11px !important; opacity: 0.8 !important; font-weight: 500; }
[data-testid="stSidebar"] .stNumberInput input, [data-testid="stSidebar"] .stSelectbox select { font-size: 12px !important; padding: 2px 6px !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.6; margin: 0.8rem 0 0.25rem; font-weight: 600 !important; }
[data-testid="stMetric"] { background: rgba(130, 130, 130, 0.08); border: 1px solid rgba(130, 130, 130, 0.2); border-radius: 8px; padding: 10px 14px !important; }
[data-testid="stMetricLabel"] { font-size: 10px !important; opacity: 0.7 !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 700 !important; }
.stCaption { font-size: 11px !important; opacity: 0.7; }
.stAlert p { font-size: 12px !important; margin: 0; }
.stTabs [data-baseweb="tab"] { font-size: 12px !important; padding: 6px 14px !important; }
.sec-label { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.6; margin: 0 0 4px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 1. ENGINEERING ENGINES (TRUE BEAM MATH)
# ============================================================
ES = 200_000
ECU = 0.003
PHI_SHEAR = 0.75

@dataclass
class RebarGroup:
    area: float
    centroid: float
    extreme_fiber: float
    width_req: float
    layers: list

def get_rebar_group(n1, dia1, n2, dia2, cover, tie_d, clr=25) -> RebarGroup:
    """Calculates properties for a group of bars (Top or Bottom)."""
    if (n1 + n2) == 0: return RebarGroup(0.0, 0.0, 0.0, 0.0, [])
    s = max(clr, 25, max(dia1, dia2))
    b_req = 2 * (cover + tie_d) + n1 * dia1 + (n1 - 1) * s if n1 > 0 else 0
    layers = []
    
    A1 = n1 * math.pi * dia1 ** 2 / 4
    y1 = cover + tie_d + dia1 / 2 if n1 > 0 else 0
    if n1 > 0: layers.append((n1, dia1, y1))
    
    A2 = n2 * math.pi * dia2 ** 2 / 4
    y2 = 0
    if n2 > 0:
        y2 = y1 + dia1 / 2 + s + dia2 / 2 if n1 > 0 else cover + tie_d + dia2 / 2
        layers.append((n2, dia2, y2))
        
    tot = A1 + A2
    yc  = (A1*y1 + A2*y2) / tot if tot > 0 else 0
    ye  = y1 if n1 > 0 else (y2 if n2 > 0 else 0)
    return RebarGroup(tot, yc, ye, b_req, layers)

def beam_flexure(b, h, d, dt, dp, fc, fy, As, Asp):
    """Rigorous Strain-Compatibility Solver for Beam Flexure"""
    if As == 0:
        return {'phi_Mn': 0, 'passes_As_min': False, 'is_ductile': False, 'converged': True, 'As_min': 0, 'c': 0, 'a': 0, 'eps_t': 0, 'phi': 0, 'Mn': 0}
    
    b1 = 0.85 if fc <= 28 else max(0.65, 0.85 - 0.05 * (fc - 28) / 7)
    Asmin = max(0.25 * math.sqrt(fc) / fy * b * d, 1.4 / fy * b * d)
    
    c = d
    conv = False
    while c > 1.0:
        a = min(b1 * c, h)
        Cc = 0.85 * fc * a * b
        
        # Compression steel strain & force
        ep = ECU * (c - dp) / c if c > 0 else 0
        fp = min(fy, max(-fy, ep * ES))
        Cs = Asp * fp
        if dp <= a and ep > 0: Cs -= Asp * 0.85 * fc # remove displaced concrete
        
        # Tension steel strain & force
        es = ECU * (d - c) / c if c > 0 else 0
        T = As * min(fy, max(-fy, es * ES))
        
        if Cc + Cs <= T: 
            conv = True
            break
        c -= 0.1
        
    Mn = (Cc * (d - a/2) + Cs * (d - dp)) / 1e6
    et = ECU * (dt - c) / c if c > 0 else 0
    
    ey = fy / ES
    if et <= ey: phi = 0.65
    elif et >= ey + 0.003: phi = 0.90
    else: phi = 0.65 + 0.25 * (et - ey) / 0.003
        
    return {
        'phi_Mn': round(phi*Mn, 1), 'is_ductile': et >= 0.004,
        'As_min': round(Asmin, 1), 'passes_As_min': As >= Asmin,
        'converged': conv, 'c': round(c, 2), 'a': round(a, 2),
        'eps_t': round(et, 4), 'phi': round(phi, 3), 'Mn': round(Mn, 1)
    }

def beam_shear(b, d, fc, fyt, cov, Vu_kN, legs, bdia):
    """ACI 318-19 Simplified Shear for Beams"""
    if d <= 0: return {'final_s': 0, 'section_fails': True, 'phi_Vc': 0, 'phi_Vn': 0, 's_exact': 0, 's_max': 0}
    
    Vu = abs(Vu_kN) * 1000
    Aleg = math.pi * bdia**2 / 4
    Av = legs * Aleg
    
    Vc = 0.17 * math.sqrt(fc) * b * d
    pVc = PHI_SHEAR * Vc
    Vsreq = max((Vu/PHI_SHEAR) - Vc, 0) if Vu > pVc/2 else 0
    
    # Check max section capacity
    if Vsreq > 0.66 * math.sqrt(fc) * b * d:
        return {'final_s': 0, 'section_fails': True, 'phi_Vc': round(pVc/1000,1), 'phi_Vn': 0, 's_exact': 0, 's_max': 0}
        
    sreq = (Av * fyt * d) / Vsreq if Vsreq > 0 else 9999
    
    # ACI Minimum shear reinforcement logic
    mrat = max(0.062 * math.sqrt(fc) * b / fyt, 0.35 * b / fyt)
    smin_av = Av / mrat if mrat > 0 else 9999
    sreq = min(sreq, smin_av) if Vu > pVc/2 else 9999
    
    # Max spacing
    smxV = min(d/4, 300) if Vsreq > 0.33 * math.sqrt(fc) * b * d else min(d/2, 600)
    sex = min(sreq, smxV)
    fs = math.floor(sex/25)*25 if Vu > pVc/2 else math.floor(smxV/25)*25
    
    Vspv = Av * fyt * d / fs if fs > 0 else 0
    pVn = PHI_SHEAR * (Vc + Vspv)
    
    return {
        'final_s': fs if fs >= 50 else 0, 'section_fails': False,
        'phi_Vc': round(pVc/1000, 1), 'phi_Vn': round(pVn/1000, 1),
        's_exact': round(sex, 1), 's_max': round(smxV, 1)
    }

def run_beam_optimizer(Mu_top, Mu_bot, Vu, b, h, fc, fy, fyt, cover, tie_d, clr):
    """Separately optimizes Top and Bottom steel for a given zone's demands."""
    bars = {'DB12':12, 'DB16':16, 'DB20':20, 'DB25':25, 'DB28':28}
    
    # Generate valid 1-layer and 2-layer groups
    valid_groups = []
    for _, d1 in bars.items():
        for n1 in range(2, 10):
            rg = get_rebar_group(n1, d1, 0, 0, cover, tie_d, clr)
            if rg.width_req <= b: valid_groups.append(rg)
            
            for _, d2 in bars.items():
                for n2 in range(2, 6):
                    rg2 = get_rebar_group(n1, d1, n2, d2, cover, tie_d, clr)
                    if rg2.width_req <= b: valid_groups.append(rg2)
                    
    valid_groups.sort(key=lambda x: x.area)
    
    best_top = valid_groups[0] # Default to minimum if no demand
    for rg in valid_groups:
        d = h - rg.centroid
        dt = h - rg.extreme_fiber
        res = beam_flexure(b, h, d, dt, 50, fc, fy, rg.area, 0)
        if res['phi_Mn'] >= Mu_top and res['passes_As_min'] and res['is_ductile']:
            best_top = rg
            break
            
    best_bot = valid_groups[0]
    for rg in valid_groups:
        d = h - rg.centroid
        dt = h - rg.extreme_fiber
        res = beam_flexure(b, h, d, dt, 50, fc, fy, rg.area, 0)
        if res['phi_Mn'] >= Mu_bot and res['passes_As_min'] and res['is_ductile']:
            best_bot = rg
            break

    # Quick Shear Opt
    s_res = beam_shear(b, h - best_bot.centroid, fc, fyt, cover, Vu, 2, tie_d)
    
    return {'top': best_top, 'bot': best_bot, 'shear': s_res}

# ============================================================
# 2. VISUALIZATION & PDF (Failsafe standard fonts/chars)
# ============================================================
def draw_section(b, h, cov, tie_d, top_rg: RebarGroup, bot_rg: RebarGroup, title="", stirrup_txt=""):
    fig, ax = plt.subplots(figsize=(3.2, 3.8), dpi=120)
    fig.patch.set_facecolor('white')
    ax.add_patch(patches.Rectangle((0,0), b, h, lw=1.5, edgecolor='#444', facecolor='#f0f0f0'))
    ax.add_patch(patches.Rectangle((cov, cov), b-2*cov, h-2*cov, lw=1.0, edgecolor='#2563eb', facecolor='none', linestyle='-'))

    def draw_bars(rg, is_top, color):
        for n_bars, dia, yd in rg.layers:
            ya = (h - yd) if is_top else yd
            if n_bars == 1:
                ax.add_patch(patches.Circle((b/2, ya), dia/2, facecolor=color, edgecolor='#222', lw=0.6, zorder=4))
            else:
                x0 = cov + tie_d + dia/2
                sp = (b - 2*(cov+tie_d) - dia) / (n_bars - 1)
                for i in range(n_bars):
                    ax.add_patch(patches.Circle((x0 + i*sp, ya), dia/2, facecolor=color, edgecolor='#222', lw=0.6, zorder=4))
            ax.text(b + 8, ya, f"{int(n_bars)}-DB{int(dia)}", fontsize=6.5, color=color, va='center', ha='left', fontweight='bold')

    draw_bars(top_rg, True,  '#dc2626')
    draw_bars(bot_rg, False, '#16a34a')

    if stirrup_txt:
        ax.text(-8, h/2, stirrup_txt, fontsize=6, color='#2563eb', ha='right', va='center', fontweight='bold', rotation=90)

    ax.text(b/2, -14, f"b={int(b)}", ha='center', fontsize=6.5, color='#555')
    ax.text(-20, h/2, f"h={int(h)}", va='center', fontsize=6.5, color='#555', rotation=90)

    ax.set_xlim(-55, b + 65)
    ax.set_ylim(-28, h + 22)
    ax.set_aspect('equal')
    ax.axis('off')
    if title: ax.set_title(title, fontsize=8, fontweight='bold', pad=4, color='#333')
    plt.tight_layout(pad=0.3)
    return fig

def draw_envelope(df_env, frame_name):
    fig = plt.figure(figsize=(7, 2.8), dpi=100)
    gs  = gridspec.GridSpec(2, 1, hspace=0.08, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.plot(df_env['Station'], df_env['M3_Max'], '#1d4ed8', lw=1.2, label='+M (Sag)')
    ax1.plot(df_env['Station'], df_env['M3_Min'], '#dc2626', lw=1.2, label='-M (Hog)')
    ax1.fill_between(df_env['Station'], df_env['M3_Min'], df_env['M3_Max'], color='#94a3b8', alpha=0.15)
    ax1.axhline(0, color='#333', lw=0.6)
    ax1.invert_yaxis() # Tension on bottom convention
    ax1.set_ylabel("M (kNm)", fontsize=7)
    ax1.tick_params(labelsize=6.5)
    ax1.legend(fontsize=6, loc='lower right')
    ax1.grid(True, ls=':', lw=0.4, alpha=0.6)
    ax1.set_title(f"Envelopes - {frame_name}", fontsize=8, fontweight='bold', pad=3)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.plot(df_env['Station'], df_env['V2_Max'], '#16a34a', lw=1.2, label='+V')
    ax2.plot(df_env['Station'], df_env['V2_Min'], '#ea580c', lw=1.2, label='-V')
    ax2.fill_between(df_env['Station'], df_env['V2_Min'], df_env['V2_Max'], color='#86efac', alpha=0.2)
    ax2.axhline(0, color='#333', lw=0.6)
    ax2.set_xlabel("Station (m)", fontsize=7)
    ax2.set_ylabel("V (kN)", fontsize=7)
    ax2.tick_params(labelsize=6.5)
    ax2.legend(fontsize=6, loc='upper right')
    ax2.grid(True, ls=':', lw=0.4, alpha=0.6)

    fig.patch.set_facecolor('white')
    plt.tight_layout(pad=0.5)
    return fig

def make_pdf(b, h, fc, fy, fyt, frame, env_img, zone_data, mode):
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_margins(14, 14, 14)
    W = 182

    pdf.set_font("Arial", 'B', 13)
    pdf.cell(W, 7, "RC Beam Design - ACI 318-19", ln=True, align='C')
    pdf.set_font("Arial", '', 8.5)
    pdf.set_text_color(120,120,120)
    pdf.cell(W, 5, f"Frame: {frame}  |  {b}x{h} mm  |  f'c={fc} MPa  fy={fy} MPa  |  Input: {mode}", ln=True, align='C')
    pdf.set_text_color(0,0,0)
    pdf.ln(3)

    if env_img and os.path.exists(env_img):
        pdf.set_font("Arial",'B',9)
        pdf.set_fill_color(237,239,243)
        pdf.cell(W, 6, "  1. Force Envelopes", ln=True, fill=True)
        pdf.image(env_img, x=20, w=150)
        pdf.set_y(pdf.get_y() + 65)

    pdf.set_font("Arial",'B',9)
    pdf.set_fill_color(237,239,243)
    pdf.cell(W, 6, "  2. Zone Capacities & Detailing", ln=True, fill=True)
    pdf.ln(2)
    
    for zone in ["Left", "Mid", "Right"]:
        d = zone_data.get(zone)
        if not d: continue
        pdf.set_font("Arial",'B',8.5)
        pdf.cell(18, 5, f"  {zone}:", border=0)
        pdf.set_font("Arial",'',8.5)
        pdf.cell(W-18, 5, f"-Mu={d['Mu_top']}  +Mu={d['Mu_bot']}  |  phiMn(top)={d['phi_Mn_top']}  phiMn(bot)={d['phi_Mn_bot']}  |  Vu={d['Vu']} kN -> {d['stirrups']}", ln=True)
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tp:
        pdf.output(tp.name)
        with open(tp.name,"rb") as f: data = f.read()
    try:
        if env_img and os.path.exists(env_img): os.remove(env_img)
        os.remove(tp.name)
    except: pass
    return data

# ============================================================
# 3. UI - HEADER & MODE TOGGLE
# ============================================================

st.markdown("<h1>🏗️ RC Beam Designer <span style='font-weight:400;color:#666;font-size:0.75rem'>ACI 318-19 · Asymmetric Flexure · Shear</span></h1>", unsafe_allow_html=True)
st.caption("Left Support (i)  ·  Midspan  ·  Right Support (j)  —  all checked in one run")
st.markdown("---")

mode_col, _ = st.columns([3, 5])
with mode_col:
    input_mode = st.radio("**Force input source**", ["📂 SAP2000 CSV", "✏️ Manual values"], horizontal=True)
use_sap = input_mode == "📂 SAP2000 CSV"

forces = {z: {'M_top':0.0, 'M_bot':0.0, 'V':0.0} for z in ["Left","Mid","Right"]}
fig_env = None
env_img_path = None
selected_frame = "Manual"
df = None

if use_sap:
    up = st.file_uploader("SAP2000 frame-forces CSV", type=["csv"], label_visibility="collapsed")
    if up is None: st.stop()
    df_raw = pd.read_csv(up)
    if 'Frame' in df_raw.columns and str(df_raw['Frame'].iloc[0]).strip().lower() == 'text': df_raw = df_raw.drop(0).reset_index(drop=True)
    for col in ['Station','P','V2','V3','T','M2','M3']:
        if col in df_raw.columns: df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    hdr_col, beam_sel_col = st.columns([2, 3])
    selected_frame = beam_sel_col.selectbox("Select beam (Frame ID)", df_raw['Frame'].unique())

    df = df_raw[df_raw['Frame'] == selected_frame].copy()
    if 'OutputCase' not in df.columns: df['OutputCase'] = "N/A"
    df['V2_abs'] = df['V2'].abs()
    beam_length  = df['Station'].max()

    dl = df[df['Station'] <= 0.15*beam_length]
    dr = df[df['Station'] >= 0.85*beam_length]
    dm = df[(df['Station'] > 0.3*beam_length) & (df['Station'] < 0.7*beam_length)]
    
    # Negative M3 = Top Tension (Hogging), Positive M3 = Bot Tension (Sagging)
    forces['Left']  = {'M_top': abs(min(dl['M3'].min(), 0)), 'M_bot': max(dl['M3'].max(), 0), 'V': dl['V2_abs'].max()}
    forces['Right'] = {'M_top': abs(min(dr['M3'].min(), 0)), 'M_bot': max(dr['M3'].max(), 0), 'V': dr['V2_abs'].max()}
    forces['Mid']   = {'M_top': abs(min(dm['M3'].min(), 0)), 'M_bot': max(df['M3'].max(), 0), 'V': dm['V2_abs'].max()}

    with st.expander("📈 Show force envelopes", expanded=True):
        df_env = df.groupby('Station').agg(M3_Max=('M3','max'), M3_Min=('M3','min'), V2_Max=('V2','max'), V2_Min=('V2','min')).reset_index()
        fig_env = draw_envelope(df_env, selected_frame)
        st.pyplot(fig_env, use_container_width=True)
else:
    st.markdown("<p class='sec-label'>Enter factored demands per zone (absolute values)</p>", unsafe_allow_html=True)
    z_cols = st.columns(3)
    zone_labels = {"Left":"Left Support (i)", "Mid": "Midspan", "Right":"Right Support (j)"}
    for col, zone in zip(z_cols, ["Left","Mid","Right"]):
        with col:
            st.markdown(f"**{zone_labels[zone]}**")
            forces[zone]['M_top'] = st.number_input("-Mu (Top Tension, kNm)", value=200.0 if zone!="Mid" else 50.0, step=10.0, min_value=0.0, key=f"mt_{zone}")
            forces[zone]['M_bot'] = st.number_input("+Mu (Bot Tension, kNm)", value=50.0 if zone!="Mid" else 180.0, step=10.0, min_value=0.0, key=f"mb_{zone}")
            forces[zone]['V'] = st.number_input("Vu (kN)", value=150.0 if zone!="Mid" else 60.0, step=10.0, min_value=0.0, key=f"v_{zone}")

# ============================================================
# 4. SECTION, MATERIALS & REBAR SIDEBAR
# ============================================================
st.markdown("---")
prop_col, rebar_col = st.columns([1, 2])

with prop_col:
    st.markdown("<p class='sec-label'>Section & materials</p>", unsafe_allow_html=True)
    pc1, pc2 = st.columns(2)
    b  = pc1.number_input("b (mm)", value=300, step=50)
    h  = pc2.number_input("h (mm)", value=600, step=50)
    fc = pc1.number_input("f'c (MPa)", value=35, step=5)
    fy = pc2.number_input("fy (MPa)", value=500, step=10)

    st.markdown("<p class='sec-label'>Stirrups & Cover</p>", unsafe_allow_html=True)
    fyt_col, bv_col, legs_col = st.columns(3)
    fyt   = fyt_col.number_input("fyt (MPa)", value=400, step=10)
    bv_opts = {'RB9':9,'DB10':10,'DB12':12}
    bar_v = bv_opts[bv_col.selectbox("Size", list(bv_opts.keys()), index=1)]
    n_legs = legs_col.number_input("Legs", min_value=2, value=2)
    cov_col, clr_col = st.columns(2)
    cover = cov_col.number_input("Cover (mm)", value=40, step=5)
    clear_space = clr_col.number_input("Clr sp (mm)", value=25, step=5)

with rebar_col:
    st.markdown("<p class='sec-label'>Zone reinforcement</p>", unsafe_allow_html=True)
    auto_opt = st.checkbox("🚀 Auto-Design Optimal Bars for Me", value=False)
    
    bar_opts = {'DB12':12,'DB16':16,'DB20':20,'DB25':25,'DB28':28,'DB32':32}
    rebar_data = {}

    if auto_opt:
        st.info("Optimizer will automatically size independent Top and Bottom layers based on +Mu and -Mu demands.")
        for zone in ["Left", "Mid", "Right"]:
            opt = run_beam_optimizer(forces[zone]['M_top'], forces[zone]['M_bot'], forces[zone]['V'], b, h, fc, fy, fyt, cover, bar_v, clear_space)
            rebar_data[zone] = {'top': opt['top'], 'bot': opt['bot'], 'shear_s': opt['shear']['final_s']}
    else:
        tabs = st.tabs(["⬅ Left Support", "↔ Midspan", "➡ Right Support"])
        for tab, zone in zip(tabs, ["Left","Mid","Right"]):
            with tab:
                tc, bc = st.columns(2)
                with tc:
                    st.markdown("**Top bars**")
                    r1a, r1b = st.columns(2)
                    t_n1 = r1a.number_input("L1 n", 0, value=4 if zone!="Mid" else 2, key=f"tn1_{zone}")
                    t_d1 = bar_opts[r1b.selectbox("L1 sz", list(bar_opts.keys()), index=3, key=f"td1_{zone}")]
                    r2a, r2b = st.columns(2)
                    t_n2 = r2a.number_input("L2 n", 0, value=0, key=f"tn2_{zone}")
                    t_d2 = bar_opts[r2b.selectbox("L2 sz", list(bar_opts.keys()), index=2, key=f"td2_{zone}")]
                with bc:
                    st.markdown("**Bottom bars**")
                    s1a, s1b = st.columns(2)
                    b_n1 = s1a.number_input("L1 n", 0, value=2 if zone!="Mid" else 4, key=f"bn1_{zone}")
                    b_d1 = bar_opts[s1b.selectbox("L1 sz", list(bar_opts.keys()), index=3, key=f"bd1_{zone}")]
                    s2a, s2b = st.columns(2)
                    b_n2 = s2a.number_input("L2 n", 0, value=0, key=f"bn2_{zone}")
                    b_d2 = bar_opts[s2b.selectbox("L2 sz", list(bar_opts.keys()), index=2, key=f"bd2_{zone}")]

                top_rg = get_rebar_group(t_n1, t_d1, t_n2, t_d2, cover, bar_v, clear_space)
                bot_rg = get_rebar_group(b_n1, b_d1, b_n2, b_d2, cover, bar_v, clear_space)
                rebar_data[zone] = {'top': top_rg, 'bot': bot_rg, 'shear_s': None}

st.markdown("---")
if not st.button("▶  Run 3-Zone Design", type="primary", use_container_width=True): st.stop()

# ============================================================
# 5. RESULTS DASHBOARD
# ============================================================
cols = st.columns(3)
pdf_zone_data = {}

for idx, zone in enumerate(["Left","Mid","Right"]):
    with cols[idx]:
        top_rg, bot_rg = rebar_data[zone]['top'], rebar_data[zone]['bot']
        st.markdown(f"### {zone}")

        if top_rg.width_req > b or bot_rg.width_req > b:
            st.error(f"🚨 Bars do not fit in {b}mm width!")
            continue

        Mu_top, Mu_bot, Vu = forces[zone]['M_top'], forces[zone]['M_bot'], forces[zone]['V']

        # Flexure Top (-M)
        d_top = h - top_rg.centroid; dt_top = h - top_rg.extreme_fiber; dp_top = bot_rg.centroid
        rf_top = beam_flexure(b, h, d_top, dt_top, dp_top, fc, fy, top_rg.area, bot_rg.area)
        DC_top = round(Mu_top / rf_top['phi_Mn'], 2) if rf_top['phi_Mn'] > 0 else 999
        
        # Flexure Bot (+M)
        d_bot = h - bot_rg.centroid; dt_bot = h - bot_rg.extreme_fiber; dp_bot = top_rg.centroid
        rf_bot = beam_flexure(b, h, d_bot, dt_bot, dp_bot, fc, fy, bot_rg.area, top_rg.area)
        DC_bot = round(Mu_bot / rf_bot['phi_Mn'], 2) if rf_bot['phi_Mn'] > 0 else 999

        with st.container():
            st.markdown(f"**Flexure: Top Steel (-M)**")
            m1, m2 = st.columns(2)
            m1.metric("φMn", f"{rf_top['phi_Mn']} kNm")
            m2.metric("D/C", f"{DC_top}", delta=f"Req {round(Mu_top,1)}", delta_color="inverse" if DC_top > 1.0 else "off")
            if not rf_top['passes_As_min']: st.warning("⚠️ Fails As,min")
            
            st.markdown(f"**Flexure: Bottom Steel (+M)**")
            m3, m4 = st.columns(2)
            m3.metric("φMn", f"{rf_bot['phi_Mn']} kNm")
            m4.metric("D/C", f"{DC_bot}", delta=f"Req {round(Mu_bot,1)}", delta_color="inverse" if DC_bot > 1.0 else "off")
            if not rf_bot['passes_As_min']: st.warning("⚠️ Fails As,min")

        # Shear
        s_res = beam_shear(b, h - bot_rg.centroid, fc, fyt, cover, Vu, n_legs, bar_v)
        final_s = rebar_data[zone]['shear_s'] if auto_opt else s_res['final_s']
        DC_v = round(Vu / s_res['phi_Vn'], 2) if s_res['phi_Vn'] > 0 else 999
        
        st.markdown(f"**Shear**")
        s1, s2 = st.columns(2)
        s1.metric("φVn", f"{s_res['phi_Vn']} kN")
        s2.metric("D/C", f"{DC_v}", delta=f"Vu {round(Vu,1)}", delta_color="inverse" if DC_v > 1.0 else "off")
        
        if final_s > 0:
            stir_lbl = f"{n_legs}-DB{bar_v} @ {final_s}"
            st.success(f"✅ {stir_lbl} mm")
        else:
            stir_lbl = "FAILS"
            st.error("❌ Shear capacity failed.")

        # Sketch
        fig_cs = draw_section(b, h, cover, bar_v, top_rg, bot_rg, zone, stir_lbl)
        st.pyplot(fig_cs, use_container_width=True)
        plt.close(fig_cs)

        pdf_zone_data[zone] = {
            'Mu_top': round(Mu_top,1), 'phi_Mn_top': rf_top['phi_Mn'],
            'Mu_bot': round(Mu_bot,1), 'phi_Mn_bot': rf_bot['phi_Mn'],
            'Vu': round(Vu,1), 'stirrups': stir_lbl
        }

if any(v is not None for v in pdf_zone_data.values()):
    st.markdown("---")
    pdf_bytes = make_pdf(b, h, fc, fy, fyt, selected_frame, env_img_path, pdf_zone_data, "SAP2000" if use_sap else "Manual")
    st.download_button("📄 Download PDF report", data=pdf_bytes, file_name=f"Beam_{selected_frame}.pdf", type="primary")
