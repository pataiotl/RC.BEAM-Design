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
st.set_page_config(
    page_title="RC Beam Designer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"] { font-size: 12.5px; }

/* ── Titles ── */
h1  { font-size: 1.25rem !important; font-weight: 700 !important; letter-spacing: -0.3px; margin-bottom: 2px !important; }
h2  { font-size: 1.0rem  !important; font-weight: 600 !important; }
h3  { font-size: 0.9rem  !important; font-weight: 600 !important; }
h4, h5 { font-size: 0.82rem !important; font-weight: 600 !important; }

/* ── Input labels shrink ── */
label { font-size: 11.5px !important; color: #444 !important; }
.stNumberInput input, .stSelectbox select { font-size: 12px !important; padding: 2px 6px !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #f8f9fb; border: 1px solid #e2e5ea;
    border-radius: 7px; padding: 8px 12px !important;
}
[data-testid="stMetricLabel"]  { font-size: 9.5px !important; color: #888 !important;
    text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricValue"]  { font-size: 1.25rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"]  { font-size: 9px !important; }

/* ── Alerts smaller ── */
.stAlert p { font-size: 11.5px !important; margin: 0; }
.stAlert    { padding: 6px 10px !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] { font-size: 11.5px !important; padding: 5px 12px !important; }

/* ── Buttons ── */
.stButton button, .stDownloadButton button {
    font-size: 12px !important; padding: 5px 14px !important; border-radius: 5px !important;
}

/* ── Expander ── */
.streamlit-expanderHeader { font-size: 11.5px !important; }

/* ── Caption ── */
.stCaption { font-size: 10.5px !important; color: #999; }

/* ── Section label helper ── */
.sec-label {
    font-size: 9.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: #aaa; margin: 10px 0 3px;
}

/* ── Zone result card ── */
.zone-card {
    border: 1px solid #e2e5ea; border-radius: 8px;
    padding: 10px 12px; background: #fff; margin-bottom: 8px;
}
.pass-badge  { color: #1a7a3c; font-weight: 700; font-size: 11px; }
.fail-badge  { color: #b91c1c; font-weight: 700; font-size: 11px; }
.warn-badge  { color: #92400e; font-weight: 700; font-size: 11px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 1. ENGINEERING ENGINES  (unchanged logic, cleaned up)
# ============================================================

@dataclass
class RebarGroup:
    area: float
    centroid: float
    extreme_fiber: float
    width_req: float
    layers: list


def get_rebar_group(n1, dia1, n2, dia2, n3, dia3, cover, tie_d, clr=25) -> RebarGroup:
    if (n1 + n2 + n3) == 0:
        return RebarGroup(0.0, 0.0, 0.0, 0.0, [])
    max_dia = max(dia1 if n1 > 0 else 0, dia2 if n2 > 0 else 0, dia3 if n3 > 0 else 0)
    s = max(clr, 25, max_dia)
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
    A3 = n3 * math.pi * dia3 ** 2 / 4
    y3 = 0
    if n3 > 0:
        if n2 > 0:   y3 = y2 + dia2 / 2 + s + dia3 / 2
        elif n1 > 0: y3 = y1 + dia1 / 2 + s + dia3 / 2
        else:        y3 = cover + tie_d + dia3 / 2
        layers.append((n3, dia3, y3))
    tot = A1 + A2 + A3
    yc  = (A1*y1 + A2*y2 + A3*y3) / tot if tot > 0 else 0
    ye  = y1 if n1 > 0 else (y2 if n2 > 0 else (y3 if n3 > 0 else 0))
    return RebarGroup(tot, yc, ye, b_req, layers)


def beam_flexure(b, h, d, dt, dp, fc, fy, As, Asp):
    if As == 0:
        return {'phi_Mn': 0, 'passes_As_min': False, 'is_ductile': False,
                'converged': True, 'As_min': 0, 'c': 0, 'a': 0, 'eps_t': 0, 'phi': 0, 'Mn': 0}
    Es  = 200_000
    ecu = 0.003
    ey  = fy / Es
    b1  = 0.85 if fc <= 28 else max(0.65, 0.85 - 0.05 * (fc - 28) / 7)
    Asmin = max(0.25 * math.sqrt(fc) / fy * b * d, 1.4 / fy * b * d)
    c = d; conv = False
    while c > 1.0:
        a  = min(b1 * c, h)
        Cc = 0.85 * fc * a * b
        ep = ecu * (c - dp) / c if c > 0 else 0
        fp = min(fy, max(-fy, ep * Es))
        Cs = Asp * fp
        if dp <= a and ep > 0: Cs -= Asp * 0.85 * fc
        es = ecu * (d - c) / c if c > 0 else 0
        T  = As * min(fy, max(-fy, es * Es))
        if Cc + Cs <= T: conv = True; break
        c -= 0.1
    Mn  = (Cc * (d - a/2) + Cs * (d - dp)) / 1e6
    et  = ecu * (dt - c) / c if c > 0 else 0
    phi = 0.65 if et <= ey else (0.90 if et >= ey + 0.003
          else 0.65 + 0.25 * (et - ey) / 0.003)
    return {'phi_Mn': round(phi*Mn,1), 'is_ductile': et >= 0.004,
            'As_min': round(Asmin,1), 'passes_As_min': As >= Asmin,
            'converged': conv, 'c': round(c,2), 'a': round(a,2),
            'eps_t': round(et,5), 'phi': round(phi,3), 'Mn': round(Mn,1)}


def shear_torsion(b, h, d, fc, fyt, fyl, cov, Vu_kN, Tu_kNm, legs, bdia, lam):
    if d <= 0:
        return {k: 0 for k in ['final_s','section_fails','needs_torsion','Al_req','Al_min',
                                's_exact','s_max','combined_stress','stress_limit',
                                'lambda_s','phi_Vc','phi_Vn','T_th']}
    pv = 0.75
    Vu = abs(Vu_kN) * 1000
    Tu = abs(Tu_kNm) * 1e6
    Aleg = math.pi * bdia**2 / 4
    ls = min(1.0, math.sqrt(2 / (1 + 0.004*d)))
    Vc = 0.17 * ls * lam * math.sqrt(fc) * b * d
    pVc = pv * Vc
    x1, y1 = b - 2*(cov+bdia/2), h - 2*(cov+bdia/2)
    Aoh = x1*y1; Ao = 0.85*Aoh; ph = 2*(x1+y1)
    Acp = b*h; pcp = 2*(b+h)
    Tth = pv * 0.083 * lam * math.sqrt(fc) * Acp**2 / pcp
    need_T = Tu > Tth
    Vsreq = max((Vu/pv) - Vc, 0) if Vu > pVc/2 else 0
    Avsreq = Vsreq / (fyt*d) if Vsreq > 0 else 0
    if need_T:
        Atreq = (Tu/pv) / (2*Ao*fyt)
        Atmin = max(Atreq, 0.175*b/fyt)
        Almin = max(0, 0.42*math.sqrt(fc)*Acp/fyl - Atmin*ph*(fyt/fyl))
        Alreq = max(Atreq*ph*(fyt/fyl), Almin)
    else:
        Atreq = Alreq = Almin = 0
    rpl  = (Avsreq/legs) + Atreq
    sc   = Aleg / rpl if rpl > 0 else 9999
    mrat = max(0.062*math.sqrt(fc)*b/fyt, 0.35*b/fyt)
    sm   = Aleg / (mrat/2) if mrat > 0 else 9999
    sreq = min(sc, sm)
    smxV = min(d/4,300) if Vsreq > 0.33*math.sqrt(fc)*b*d else min(d/2,600)
    smax = min(smxV, min(ph/8,300)) if need_T else smxV
    sex  = min(sreq, smax)
    fs   = math.floor(sex/25)*25
    Vspv = legs * Aleg * fyt * d / fs if fs > 0 else 0
    pVn  = pv * (Vc + Vspv)
    vs   = Vu/(b*d)
    ts   = (Tu*ph)/(1.7*Aoh**2) if need_T else 0
    cs   = math.sqrt(vs**2 + ts**2)
    sl   = pv * (Vc/(b*d) + 0.66*math.sqrt(fc))
    fail = cs > sl
    return {'final_s': 0 if fs < 50 or fail else fs,
            'section_fails': fail, 'needs_torsion': need_T,
            'Al_req': round(Alreq,1), 'Al_min': round(Almin,1),
            'T_th': round(Tth/1e6,1), 'phi_Vc': round(pVc/1000,1),
            'phi_Vn': round(pVn/1000,1), 'lambda_s': round(ls,3),
            's_exact': round(sex,1), 's_max': round(smax,1),
            'combined_stress': round(cs,2), 'stress_limit': round(sl,2)}


def dev_length(db, fy, fc, top, cov, clr, lam):
    if db == 0: return {'ld': 0, 'lap': 0, 'ldh': 0}
    pt = 1.3 if top else 1.0
    ps = 0.8 if db <= 20 else 1.0
    cb = min(cov + db/2, clr/2 + db/2)
    ct = min(cb/db, 2.5)
    ld = max((fy / (1.1*lam*math.sqrt(fc))) * (pt*ps/ct) * db, 300)
    lp = max(1.3*ld, 300)
    lh = max(0.24*fy/(lam*math.sqrt(fc))*db, 8*db, 150) * (1.2 if db > 20 else 1.0)
    return {'ld': math.ceil(ld/50)*50, 'lap': math.ceil(lp/50)*50, 'ldh': math.ceil(lh/50)*50}


# ============================================================
# 2. VISUALISATION — compact cross-section drawing
# ============================================================

def draw_section(b, h, cov, tie_d, top_rg: RebarGroup, bot_rg: RebarGroup,
                 title="", stirrup_txt=""):
    fig, ax = plt.subplots(figsize=(3.2, 3.8), dpi=120)
    fig.patch.set_facecolor('white')
    # Concrete outline
    ax.add_patch(patches.Rectangle((0,0), b, h, lw=1.5,
                 edgecolor='#444', facecolor='#f0f0f0'))
    # Cover box (stirrup)
    ax.add_patch(patches.Rectangle((cov, cov), b-2*cov, h-2*cov,
                 lw=1.0, edgecolor='#2563eb', facecolor='none', linestyle='-'))

    def draw_bars(rg, is_top, color):
        for n_bars, dia, yd in rg.layers:
            ya = (h - yd) if is_top else yd
            if n_bars == 1:
                ax.add_patch(patches.Circle((b/2, ya), dia/2,
                             facecolor=color, edgecolor='#222', lw=0.6, zorder=4))
            else:
                x0 = cov + tie_d + dia/2
                sp = (b - 2*(cov+tie_d) - dia) / (n_bars - 1)
                for i in range(n_bars):
                    ax.add_patch(patches.Circle((x0 + i*sp, ya), dia/2,
                                 facecolor=color, edgecolor='#222', lw=0.6, zorder=4))
            # Label right side
            label = f"{int(n_bars)}-Ø{int(dia)}"
            ax.text(b + 8, ya, label, fontsize=6.5, color=color,
                    va='center', ha='left', fontweight='bold')

    draw_bars(top_rg, True,  '#dc2626')
    draw_bars(bot_rg, False, '#16a34a')

    # Stirrup annotation on left
    if stirrup_txt:
        ax.text(-8, h/2, stirrup_txt, fontsize=6, color='#2563eb',
                ha='right', va='center', fontweight='bold',
                rotation=90)

    # Dimensions
    ax.text(b/2, -14, f"b={int(b)}", ha='center', fontsize=6.5, color='#555')
    ax.text(-20, h/2, f"h={int(h)}", va='center', fontsize=6.5, color='#555', rotation=90)

    ax.set_xlim(-55, b + 65)
    ax.set_ylim(-28, h + 22)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=8, fontweight='bold', pad=4, color='#333')
    plt.tight_layout(pad=0.3)
    return fig


# ============================================================
# 3. ENVELOPE PLOT — compact 2-panel
# ============================================================

def draw_envelope(df_env, frame_name):
    fig = plt.figure(figsize=(7, 2.8), dpi=100)
    gs  = gridspec.GridSpec(2, 1, hspace=0.08, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.plot(df_env['Station'], df_env['M3_Max'], '#1d4ed8', lw=1.2, label='+M')
    ax1.plot(df_env['Station'], df_env['M3_Min'], '#dc2626', lw=1.2, label='−M')
    ax1.fill_between(df_env['Station'], df_env['M3_Min'], df_env['M3_Max'],
                     color='#94a3b8', alpha=0.15)
    ax1.axhline(0, color='#333', lw=0.6)
    ax1.invert_yaxis()
    ax1.set_ylabel("M (kNm)", fontsize=7)
    ax1.tick_params(labelsize=6.5)
    ax1.legend(fontsize=6, loc='lower right', framealpha=0.7, handlelength=1.2)
    ax1.grid(True, ls=':', lw=0.4, alpha=0.6)
    ax1.set_title(f"Envelopes — {frame_name}", fontsize=8, fontweight='bold', pad=3)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.plot(df_env['Station'], df_env['V2_Max'], '#16a34a', lw=1.2, label='+V')
    ax2.plot(df_env['Station'], df_env['V2_Min'], '#ea580c', lw=1.2, label='−V')
    ax2.fill_between(df_env['Station'], df_env['V2_Min'], df_env['V2_Max'],
                     color='#86efac', alpha=0.2)
    ax2.axhline(0, color='#333', lw=0.6)
    ax2.set_xlabel("Station (m)", fontsize=7)
    ax2.set_ylabel("V (kN)", fontsize=7)
    ax2.tick_params(labelsize=6.5)
    ax2.legend(fontsize=6, loc='upper right', framealpha=0.7, handlelength=1.2)
    ax2.grid(True, ls=':', lw=0.4, alpha=0.6)

    fig.patch.set_facecolor('white')
    plt.tight_layout(pad=0.5)
    return fig


# ============================================================
# 4. PDF REPORT
# ============================================================

def make_pdf(b, h, fc, fy, fyt, frame, env_img, zone_data, mode):
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_margins(14, 14, 14)
    pdf.set_auto_page_break(auto=False)
    W = 182

    pdf.set_font("Arial", 'B', 13)
    pdf.cell(W, 7, "RC Beam Design — ACI 318-19", ln=True, align='C')
    pdf.set_font("Arial", '', 8.5)
    pdf.set_text_color(120,120,120)
    pdf.cell(W, 5, f"Frame: {frame}  |  {b}×{h} mm  |  f'c={fc} MPa  fy={fy} MPa  fyt={fyt} MPa  |  Input: {mode}",
             ln=True, align='C')
    pdf.set_text_color(0,0,0)
    pdf.ln(3)

    def sec(t):
        pdf.set_font("Arial",'B',9)
        pdf.set_fill_color(237,239,243)
        pdf.cell(W, 6, f"  {t}", ln=True, fill=True)
        pdf.set_font("Arial",'',8.5)
        pdf.ln(0.5)

    sec("1. Force Envelopes / Demands")
    if env_img and os.path.exists(env_img):
        yb = pdf.get_y()
        pdf.image(env_img, x=20, w=150)
        pdf.set_y(yb + 60)
    else:
        pdf.cell(W, 5, "  (Manual input — no envelope)", ln=True)
    pdf.ln(2)

    sec("2. Zone Capacities & Detailing")
    for zone in ["Left","Mid","Right"]:
        d = zone_data.get(zone)
        if not d: continue
        pdf.set_font("Arial",'B',8.5)
        pdf.cell(18, 5, f"  {zone}:", border=0)
        pdf.set_font("Arial",'',8.5)
        pdf.cell(W-18, 5,
            f"Mu={d['Mu']} kNm  φMn={d['phi_Mn']} kNm  D/C={d['DC_flex']}  |  "
            f"Vu={d['Vu']} kN  →  {d['stirrups']}", ln=True)
        pdf.cell(18, 5, "", border=0)
        if zone in ["Left","Right"]:
            pdf.cell(W-18, 5,
                f"Top hook ldh={d['dev_top']} mm  |  Top lap={d['dev_top_lap']} mm", ln=True)
        else:
            pdf.cell(W-18, 5, f"Bot lap splice={d['dev_bot']} mm", ln=True)
        pdf.ln(0.5)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tp:
        pdf.output(tp.name)
        with open(tp.name,"rb") as f: data = f.read()
    try:
        if env_img and os.path.exists(env_img): os.remove(env_img)
        os.remove(tp.name)
    except: pass
    return data


# ============================================================
# 5. UI — HEADER
# ============================================================

st.markdown(
    "<h1>🏗️ RC Beam Designer <span style='font-weight:400;color:#666;font-size:0.75rem'>"
    "ACI 318-19 · 3-Zone · Flexure · Shear · Torsion · Dev. Length</span></h1>",
    unsafe_allow_html=True
)
st.caption("Left Support (i)  ·  Midspan  ·  Right Support (j)  —  all checked in one run")
st.markdown("---")

# ============================================================
# 6. INPUT MODE TOGGLE
# ============================================================

mode_col, _ = st.columns([3, 5])
with mode_col:
    input_mode = st.radio(
        "**Force input source**",
        ["📂 SAP2000 CSV", "✏️ Manual values"],
        horizontal=True,
        label_visibility="visible"
    )
use_sap = input_mode == "📂 SAP2000 CSV"

# Shared state
forces = {z: {'M':0.0,'V':0.0,'T':0.0} for z in ["Left","Mid","Right"]}
beam_length = 0.0
fig_env = None
env_img_path = None
selected_frame = "Manual"
df = None
max_V2_raw = 0.0; combo_V2 = "—"; stat_V2 = 0.0

# ============================================================
# PATH A — SAP2000
# ============================================================
if use_sap:
    up = st.file_uploader("SAP2000 frame-forces CSV",
                           type=["csv"], label_visibility="collapsed")
    if up is None:
        st.info("Upload a SAP2000 frame-forces CSV.  Required columns: Frame, OutputCase, Station, V2, T, M3.")
        st.stop()

    df_raw = pd.read_csv(up)
    if 'Frame' in df_raw.columns and str(df_raw['Frame'].iloc[0]).strip().lower() == 'text':
        df_raw = df_raw.drop(0).reset_index(drop=True)
    for col in ['Station','P','V2','V3','T','M2','M3']:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    if 'Frame' not in df_raw.columns:
        st.error("CSV must contain a 'Frame' column.")
        st.stop()

    hdr_col, beam_sel_col = st.columns([2, 3])
    with beam_sel_col:
        selected_frame = st.selectbox("Select beam (Frame ID)", df_raw['Frame'].unique())

    df = df_raw[df_raw['Frame'] == selected_frame].copy()
    if 'OutputCase' not in df.columns: df['OutputCase'] = "N/A"
    df['V2_abs'] = df['V2'].abs()
    df['T_abs']  = df['T'].abs()
    beam_length  = df['Station'].max()

    # Zone extraction
    dl = df[df['Station'] <= 0.1*beam_length]
    dr = df[df['Station'] >= 0.9*beam_length]
    dm = df[(df['Station'] > 0.3*beam_length) & (df['Station'] < 0.7*beam_length)]
    forces['Left']  = {'M': dl['M3'].min() if not dl.empty else 0,
                        'V': dl['V2_abs'].max() if not dl.empty else 0,
                        'T': dl['T_abs'].max() if not dl.empty else 0}
    forces['Right'] = {'M': dr['M3'].min() if not dr.empty else 0,
                        'V': dr['V2_abs'].max() if not dr.empty else 0,
                        'T': dr['T_abs'].max() if not dr.empty else 0}
    forces['Mid']   = {'M': df['M3'].max(),
                        'V': dm['V2_abs'].max() if not dm.empty else 0,
                        'T': dm['T_abs'].max() if not dm.empty else 0}

    # Key demand metrics in a compact strip
    idx_pM = df['M3'].idxmax(); idx_nM = df['M3'].idxmin()
    idx_V  = df['V2_abs'].idxmax(); idx_T = df['T_abs'].idxmax()
    max_V2_raw = df.loc[idx_V,'V2_abs']
    combo_V2   = df.loc[idx_V,'OutputCase']
    stat_V2    = df.loc[idx_V,'Station']

    mc1,mc2,mc3,mc4,mc5 = st.columns(5)
    mc1.metric("Beam", selected_frame)
    mc2.metric("Max +M", f"{round(df.loc[idx_pM,'M3'],1)} kNm",
               f"{df.loc[idx_pM,'OutputCase']}")
    mc3.metric("Max −M", f"{round(abs(df.loc[idx_nM,'M3']),1)} kNm",
               f"{df.loc[idx_nM,'OutputCase']}")
    mc4.metric("Max V2", f"{round(max_V2_raw,1)} kN",
               f"@ {stat_V2} m")
    mc5.metric("Max T",  f"{round(df.loc[idx_T,'T_abs'],1)} kNm",
               f"{df.loc[idx_T,'OutputCase']}")

    # Deflection note
    h_min = beam_length * 1000 / 18.5
    st.caption(f"ACI h_min (L/18.5, one-end continuous) ≈ {round(h_min,0):.0f} mm — adjust for actual end conditions")

    # Envelope chart — placed compactly beside metrics using expander
    with st.expander("📈 Show force envelopes", expanded=True):
        df_env = df.groupby('Station').agg(
            M3_Max=('M3','max'), M3_Min=('M3','min'),
            V2_Max=('V2','max'), V2_Min=('V2','min')
        ).reset_index()
        fig_env = draw_envelope(df_env, selected_frame)
        st.pyplot(fig_env, use_container_width=True)

# ============================================================
# PATH B — MANUAL
# ============================================================
else:
    st.markdown("<p class='sec-label'>Enter factored demands per zone (absolute values)</p>",
                unsafe_allow_html=True)

    beam_length = st.number_input("Beam span L (m) — for h_min note only",
                                  value=6.0, step=0.5, min_value=1.0,
                                  label_visibility="visible")
    h_min = beam_length * 1000 / 18.5
    st.caption(f"ACI h_min ≈ {round(h_min,0):.0f} mm (L/18.5, one-end continuous)")

    z_cols = st.columns(3)
    zone_labels = {"Left":"Left Support (i)  —  top bars in tension",
                   "Mid": "Midspan  —  bottom bars in tension",
                   "Right":"Right Support (j)  —  top bars in tension"}
    for col, zone in zip(z_cols, ["Left","Mid","Right"]):
        with col:
            st.markdown(f"**{zone_labels[zone]}**")
            forces[zone]['M'] = st.number_input(
                "Mu (kNm)", value=200.0 if zone=="Left" else (180.0 if zone=="Mid" else 220.0),
                step=5.0, min_value=0.0, key=f"m_{zone}")
            forces[zone]['V'] = st.number_input(
                "Vu (kN)", value=150.0 if zone=="Left" else (60.0 if zone=="Mid" else 155.0),
                step=5.0, min_value=0.0, key=f"v_{zone}")
            forces[zone]['T'] = st.number_input(
                "Tu (kNm)", value=0.0, step=1.0, min_value=0.0, key=f"t_{zone}")
    selected_frame = "Manual"

# ============================================================
# 7. SECTION, MATERIALS & REBAR — compact 3-column layout
# ============================================================

st.markdown("---")

# ── Row 1: Section props + Stirrups (left column), Rebar input (right 2 columns)
prop_col, rebar_col = st.columns([1, 2])

with prop_col:
    st.markdown("<p class='sec-label'>Section & materials</p>", unsafe_allow_html=True)
    pc1, pc2 = st.columns(2)
    b  = pc1.number_input("b (mm)", value=300, step=50, min_value=150)
    h  = pc2.number_input("h (mm)", value=600, step=50, min_value=200)
    fc_col, lam_col = st.columns(2)
    fc       = fc_col.number_input("f'c (MPa)", value=35, step=5, min_value=20)
    lambda_c = lam_col.selectbox("λ", [1.0, 0.85, 0.75],
                                  format_func=lambda x: {1.0:"1.0 NW",0.85:"0.85 SLW",0.75:"0.75 LW"}[x])

    st.markdown("<p class='sec-label'>Stirrups</p>", unsafe_allow_html=True)
    fyt_col, bv_col, legs_col = st.columns(3)
    fyt   = fyt_col.number_input("fyt (MPa)", value=400, step=10, min_value=240)
    bv_opts = {'RB9':9,'DB10':10,'DB12':12,'DB16':16}
    bar_v = bv_opts[bv_col.selectbox("Size", list(bv_opts.keys()), index=1)]
    n_legs = legs_col.number_input("Legs", min_value=2, value=2, step=1)

    cov_col, clr_col = st.columns(2)
    cover_clear = cov_col.number_input("Cover (mm)", value=40, step=5, min_value=20)
    clear_space = clr_col.number_input("Clr sp (mm)", value=25, step=5, min_value=20)

    st.markdown("<p class='sec-label'>Main steel</p>", unsafe_allow_html=True)
    fy = st.number_input("fy (MPa)", value=500, step=10, min_value=300)

with rebar_col:
    st.markdown("<p class='sec-label'>Zone reinforcement — specify top & bottom bars per zone</p>",
                unsafe_allow_html=True)
    bar_opts = {'DB12':12,'DB16':16,'DB20':20,'DB25':25,'DB28':28,'DB32':32}
    tabs = st.tabs(["⬅ Left Support (i)", "↔ Midspan", "➡ Right Support (j)"])
    rebar_data = {}
    bar_sel    = {}

    for tab, zone in zip(tabs, ["Left","Mid","Right"]):
        with tab:
            def_t = 4 if zone in ["Left","Right"] else 2
            def_b = 4 if zone == "Mid" else 2
            tc, bc = st.columns(2)

            with tc:
                st.markdown("**Top bars** *(compression at midspan / tension at supports)*")
                r1a, r1b = st.columns(2)
                t_n1 = r1a.number_input("L1 n", 0, value=def_t, key=f"tn1_{zone}")
                t_d1 = bar_opts[r1b.selectbox("L1 sz", list(bar_opts.keys()), index=3, key=f"td1_{zone}")]
                r2a, r2b = st.columns(2)
                t_n2 = r2a.number_input("L2 n", 0, value=0, key=f"tn2_{zone}")
                t_d2 = bar_opts[r2b.selectbox("L2 sz", list(bar_opts.keys()), index=2, key=f"td2_{zone}")]
                r3a, r3b = st.columns(2)
                t_n3 = r3a.number_input("L3 n", 0, value=0, key=f"tn3_{zone}")
                t_d3 = bar_opts[r3b.selectbox("L3 sz", list(bar_opts.keys()), index=2, key=f"td3_{zone}")]

            with bc:
                st.markdown("**Bottom bars** *(tension at midspan / compression at supports)*")
                s1a, s1b = st.columns(2)
                b_n1 = s1a.number_input("L1 n", 0, value=def_b, key=f"bn1_{zone}")
                b_d1 = bar_opts[s1b.selectbox("L1 sz", list(bar_opts.keys()), index=3, key=f"bd1_{zone}")]
                s2a, s2b = st.columns(2)
                b_n2 = s2a.number_input("L2 n", 0, value=0, key=f"bn2_{zone}")
                b_d2 = bar_opts[s2b.selectbox("L2 sz", list(bar_opts.keys()), index=2, key=f"bd2_{zone}")]
                s3a, s3b = st.columns(2)
                b_n3 = s3a.number_input("L3 n", 0, value=0, key=f"bn3_{zone}")
                b_d3 = bar_opts[s3b.selectbox("L3 sz", list(bar_opts.keys()), index=2, key=f"bd3_{zone}")]

            top_rg = get_rebar_group(t_n1,t_d1,t_n2,t_d2,t_n3,t_d3, cover_clear, bar_v, clear_space)
            bot_rg = get_rebar_group(b_n1,b_d1,b_n2,b_d2,b_n3,b_d3, cover_clear, bar_v, clear_space)
            rebar_data[zone] = {'top': top_rg, 'bot': bot_rg}
            bar_sel[zone]    = {'top_d': t_d1 if t_n1>0 else 0,
                                 'bot_d': b_d1 if b_n1>0 else 0}

            # Live area preview
            st.caption(
                f"Top Ast = {round(top_rg.area,0):.0f} mm²  ·  "
                f"Bot Ast = {round(bot_rg.area,0):.0f} mm²  ·  "
                f"d (top governs) ≈ {round(h - top_rg.centroid,0):.0f} mm  "
                f"d (bot governs) ≈ {round(h - bot_rg.centroid,0):.0f} mm"
            )

# ============================================================
# 8. RUN BUTTON
# ============================================================

st.markdown("---")
run = st.button("▶  Run 3-Zone Design", type="primary", use_container_width=True)

if not run:
    st.stop()

# ============================================================
# 9. RESULTS — 3-column zone cards
# ============================================================

if fig_env is not None:
    tmp_env = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig_env.savefig(tmp_env.name, bbox_inches='tight', dpi=130)
    env_img_path = tmp_env.name

cols   = st.columns(3)
zones  = ["Left","Mid","Right"]
zone_icons = {"Left":"⬅","Mid":"↔","Right":"➡"}
pdf_zone_data = {}

for idx, zone in enumerate(zones):
    with cols[idx]:
        top_rg = rebar_data[zone]['top']
        bot_rg = rebar_data[zone]['bot']

        # ── Header ──────────────────────────────────
        st.markdown(f"### {zone_icons[zone]} {zone}")

        # ── Bar fit ─────────────────────────────────
        if top_rg.width_req > b or bot_rg.width_req > b:
            st.error(f"🚨 Bars don't fit!  "
                     f"Top needs {round(top_rg.width_req,0):.0f} mm, "
                     f"Bot needs {round(bot_rg.width_req,0):.0f} mm")
            pdf_zone_data[zone] = None
            continue

        # ── Effective depths ─────────────────────────
        if zone == "Mid":
            As_t, As_c = bot_rg.area, top_rg.area
            d = h - bot_rg.centroid; dt = h - bot_rg.extreme_fiber; dp = top_rg.centroid
        else:
            As_t, As_c = top_rg.area, bot_rg.area
            d = h - top_rg.centroid; dt = h - top_rg.extreme_fiber; dp = bot_rg.centroid

        Mu = abs(forces[zone]['M'])

        # ── FLEXURE ──────────────────────────────────
        rf = beam_flexure(b, h, d, dt, dp, fc, fy, As_t, As_c)
        DC_f = round(Mu / rf['phi_Mn'], 2) if rf['phi_Mn'] > 0 else 999.9

        # Compact result line
        flex_ok = rf['converged'] and rf['passes_As_min'] and rf['is_ductile'] and rf['phi_Mn'] >= Mu
        flex_icon = "✅" if flex_ok else ("⚠️" if not rf['passes_As_min'] else "❌")

        with st.container():
            st.markdown(f"**Flexure**")
            m1, m2 = st.columns(2)
            m1.metric("φMn", f"{rf['phi_Mn']} kNm")
            m2.metric("D/C", f"{DC_f}", delta=f"Req {round(Mu,1)} kNm",
                      delta_color="inverse" if DC_f > 1.0 else "off")

        if not rf['converged']:
            st.error("🚨 Solver failed")
        elif not rf['passes_As_min']:
            st.warning(f"⚠️ As < As,min ({rf['As_min']} mm²)")
        elif not rf['is_ductile']:
            st.error("❌ Over-reinforced")
        elif rf['phi_Mn'] >= Mu:
            st.success(f"✅ Flexure OK  (ε_t={rf['eps_t']}, φ={rf['phi']})")
        else:
            st.error("❌ Flexure fails")

        # ── SHEAR ────────────────────────────────────
        if use_sap and df is not None and beam_length > 0 and zone in ["Left","Right"]:
            dm = d / 1000.0
            valid = df[(df['Station'] >= dm) & (df['Station'] <= beam_length - dm)]
            if not valid.empty:
                sub = valid[valid['Station'] <= 0.3*beam_length] if zone=="Left" \
                      else valid[valid['Station'] >= 0.7*beam_length]
                iv  = sub['V2_abs'].idxmax() if not sub.empty else valid['V2_abs'].idxmax()
                Vu_d = valid.loc[iv,'V2_abs']
                c_lbl = f"{valid.loc[iv,'OutputCase']} @ {round(valid.loc[iv,'Station'],2)} m"
            else:
                Vu_d = max_V2_raw; c_lbl = combo_V2
        else:
            Vu_d  = forces[zone]['V']
            c_lbl = "manual"

        rs = shear_torsion(b, h, d, fc, fyt, fy, cover_clear,
                           Vu_d, forces[zone]['T'], n_legs, bar_v, lambda_c)
        pVn   = rs['phi_Vn']
        DC_sh = round(Vu_d / pVn, 2) if pVn > 0 else 999.9

        st.markdown(f"**Shear**  <span style='color:#666;font-size:10px'>({c_lbl})</span>",
                    unsafe_allow_html=True)
        sh1, sh2 = st.columns(2)
        sh1.metric("φVn", f"{pVn} kN")
        sh2.metric("D/C", f"{DC_sh}", delta=f"Vu {round(Vu_d,1)} kN",
                   delta_color="inverse" if DC_sh > 1.0 else "off")

        if rs['final_s'] > 0:
            st.success(f"✅ {n_legs}-DB{bar_v} @ **{rs['final_s']} mm**"
                       f"  (req {rs['s_exact']} mm, max {rs['s_max']} mm)")
            if rs['needs_torsion']:
                st.info(f"Tu governs — add Al = {rs['Al_req']} mm² (min {rs['Al_min']} mm²)")
        elif rs['section_fails']:
            st.error(f"❌ Section too small  ({rs['combined_stress']} > {rs['stress_limit']} MPa)")
        else:
            st.error("❌ Spacing < 50 mm — increase stirrup size or legs")

        # ── DEVELOPMENT LENGTHS ──────────────────────
        dt_res = dev_length(bar_sel[zone]['top_d'], fy, fc, True,  cover_clear, clear_space, lambda_c)
        db_res = dev_length(bar_sel[zone]['bot_d'], fy, fc, False, cover_clear, clear_space, lambda_c)

        if zone in ["Left","Right"]:
            st.markdown(f"**Dev. length** — Top: ldh = **{dt_res['ldh']}** mm  ·  lap = **{dt_res['lap']}** mm")
        else:
            st.markdown(f"**Dev. length** — Bot lap = **{db_res['lap']}** mm")

        # ── CROSS-SECTION SKETCH ─────────────────────
        stir_lbl = f"{n_legs}-DB{bar_v}@{rs['final_s']}" if rs['final_s'] > 0 else "FAILS"
        fig_cs = draw_section(b, h, cover_clear, bar_v, top_rg, bot_rg, zone, stir_lbl)
        st.pyplot(fig_cs, use_container_width=True)
        plt.close(fig_cs)

        # ── CALC STEPS EXPANDER ──────────────────────
        with st.expander("🧮 Calc steps"):
            st.markdown(f"""
| Item | Value |
|------|-------|
| d / dt / d' | {round(d,1)} / {round(dt,1)} / {round(dp,1)} mm |
| c / a | {rf['c']} / {rf['a']} mm |
| ε_t | {rf['eps_t']}  →  φ = {rf['phi']} |
| Mn / φMn | {rf['Mn']} / {rf['phi_Mn']} kNm |
| λs | {rs['lambda_s']} |
| φVc / φVn | {rs['phi_Vc']} / {pVn} kN |
| s_req / s_max | {rs['s_exact']} / {rs['s_max']} mm |
| Stress check | {rs['combined_stress']} MPa ≤ {rs['stress_limit']} MPa |
""")

        # Save for PDF
        tmp_cs = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig_cs2 = draw_section(b, h, cover_clear, bar_v, top_rg, bot_rg, zone, stir_lbl)
        fig_cs2.savefig(tmp_cs.name, bbox_inches='tight', dpi=130)
        plt.close(fig_cs2)

        pdf_zone_data[zone] = {
            'Mu': round(Mu,1), 'phi_Mn': rf['phi_Mn'], 'DC_flex': DC_f,
            'Vu': round(Vu_d,1), 'DC_shear': DC_sh,
            'stirrups': (f"{n_legs}-DB{bar_v} @ {rs['final_s']} mm"
                         if rs['final_s'] > 0 else "FAILS"),
            'dev_top': dt_res['ldh'], 'dev_top_lap': dt_res['lap'],
            'dev_bot': db_res['lap'],
        }

# ============================================================
# 10. PDF EXPORT
# ============================================================

if any(v is not None for v in pdf_zone_data.values()):
    st.markdown("---")
    pdf_bytes = make_pdf(b, h, fc, fy, fyt, selected_frame,
                         env_img_path, pdf_zone_data,
                         "SAP2000 CSV" if use_sap else "Manual Input")
    dl_col, _ = st.columns([2, 5])
    dl_col.download_button(
        "📄 Download PDF report",
        data=pdf_bytes,
        file_name=f"Beam_{selected_frame}.pdf",
        mime="application/pdf",
        type="primary"
    )
