import math
import os
import tempfile
from dataclasses import dataclass

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import streamlit as st
from fpdf import FPDF


st.set_page_config(page_title="RC Beam Designer - ACI 318-19", layout="wide")

st.markdown(
    """
<style>
:root {
    --bg: #0f1117;
    --surface: #181c24;
    --surface2: #1e2330;
    --border: #2a3044;
    --text: #e8eaf0;
    --muted: #98a2b8;
    --accent: #4f8ef7;
    --pass: #22c55e;
    --fail: #f87171;
    --warn: #fbbf24;
    --info: #38bdf8;
    --torsion: #c084fc;
}
.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(79,142,247,.09), transparent 28%),
        radial-gradient(circle at 100% 8%, rgba(192,132,252,.08), transparent 22%),
        var(--bg);
    color: var(--text);
}
section[data-testid="stSidebar"], div[data-testid="stExpander"] {
    background-color: var(--surface);
}
h1, h2, h3 { letter-spacing: -0.02em; }
div[data-testid="metric-container"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 5px 6px;
    background: rgba(24,28,36,.95);
}
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 5.25px !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 12px !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 5.25px !important;
}
.app-hero {
    border: 1px solid var(--border);
    background: rgba(24,28,36,.95);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 16px;
}
.app-hero-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--text);
}
.app-hero-sub {
    font-size: .88rem;
    color: var(--muted);
    margin-top: 4px;
}
.status-card {
    border-radius: 12px;
    padding: 7px 10px;
    font-weight: 750;
    border: 1px solid;
    margin: 6px 0 10px 0;
    font-size: 9px;
}
.status-pass { color: var(--pass); background: #052a14; border-color: #166534; }
.status-fail { color: var(--fail); background: #2a0a0a; border-color: #991b1b; }
.status-warn { color: var(--warn); background: #2a1f00; border-color: #92400e; }
.check-row {
    display: grid;
    grid-template-columns: 34% minmax(0, 1fr) 62px;
    gap: 8px;
    align-items: center;
    background: rgba(24,28,36,.95);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 8px 9px;
    margin-bottom: 5px;
    font-size: 14px;
}
.check-label { font-weight: 800; color: var(--text); }
.check-detail {
    color: var(--muted);
    font-family: monospace;
    font-size: 12px;
    min-width: 0;
    overflow-wrap: anywhere;
}
.badge {
    text-align: center;
    border-radius: 5px;
    padding: 4px 6px;
    font-family: monospace;
    font-weight: 800;
    font-size: 11px;
    justify-self: end;
    min-width: 48px;
}
.badge-pass { color: var(--pass); background: #052a14; border: 1px solid #166534; }
.badge-fail { color: var(--fail); background: #2a0a0a; border: 1px solid #991b1b; }
.badge-warn { color: var(--warn); background: #2a1f00; border: 1px solid #92400e; }
.section-band {
    color: var(--muted);
    font-size: 8px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 20px 0 12px 0;
}
.zone-title {
    font-size: 18px;
    font-weight: 800;
    color: var(--text);
    margin: 6px 0 2px;
}
.zone-panel {
    border: 1px solid #2a3044;
    border-radius: 4px;
    padding: 6px;
    background: rgba(15,17,23,.55);
}
.mini-metric {
    padding: 2px 2px 5px;
    min-height: 46px;
}
.mini-metric-label {
    color: var(--text);
    font-size: 5.5px;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 3px;
}
.mini-metric-value {
    color: white;
    font-size: 12px;
    line-height: 1.05;
    font-weight: 500;
    white-space: nowrap;
}
.mini-metric-delta {
    display: inline-block;
    margin-top: 5px;
    color: var(--pass);
    background: #064e24;
    border-radius: 999px;
    padding: 2px 5px;
    font-size: 5.5px;
    font-weight: 800;
    white-space: nowrap;
}
.notice {
    background: rgba(24,28,36,.95);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 13px;
    color: var(--muted);
    line-height: 1.55;
}
@media (max-width: 700px) {
    .check-row { grid-template-columns: 1fr; }
}
</style>
    """,
    unsafe_allow_html=True,
)


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
    widths = [
        2 * (cover_clear + tie_dia) + n * dia + (n - 1) * eff_clear_space
        for n, dia in [(n1, dia1), (n2, dia2), (n3, dia3)]
        if n > 0
    ]

    layers = []
    areas = []
    ys = []
    prev_y = None
    prev_dia = None
    for n, dia in [(n1, dia1), (n2, dia2), (n3, dia3)]:
        area = n * math.pi * dia**2 / 4 if n > 0 else 0
        if n > 0:
            if prev_y is None:
                y = cover_clear + tie_dia + dia / 2
            else:
                y = prev_y + prev_dia / 2 + eff_clear_space + dia / 2
            layers.append((n, dia, y))
            prev_y, prev_dia = y, dia
        else:
            y = 0
        areas.append(area)
        ys.append(y)

    total_area = sum(areas)
    y_centroid = sum(a * y for a, y in zip(areas, ys)) / total_area if total_area else 0
    y_extreme = layers[0][2] if layers else 0
    return RebarGroup(total_area, y_centroid, y_extreme, max(widths) if widths else 0, layers)


def calculate_beam_flexure(b, h, d, dt, d_prime, fc, fy, As, As_prime):
    if As <= 0:
        return {
            "phi_Mn": 0,
            "passes_As_min": False,
            "is_ductile": False,
            "converged": True,
            "As_min": 0,
            "As_max_tc": 0,
            "passes_As_max_tc": False,
            "strain_class": "N/A",
            "c": 0,
            "a": 0,
            "eps_t": 0,
            "phi": 0,
            "Mn": 0,
        }

    Es = 200000
    ecu = 0.003
    eps_y = fy / Es
    tension_control_limit = eps_y + 0.003
    beta1 = 0.85 if fc <= 28 else max(0.65, 0.85 - 0.05 * ((fc - 28) / 7))
    As_min = max((0.25 * math.sqrt(fc) / fy) * b * d, (1.4 / fy) * b * d)
    rho_max_tc = 0.85 * beta1 * (fc / fy) * (ecu / (ecu + tension_control_limit))
    As_max_tc = rho_max_tc * b * d

    c = max(d, 1.0)
    converged = False
    a = Cc = Cs = 0
    while c > 1.0:
        a = min(beta1 * c, h)
        Cc = 0.85 * fc * a * b
        eps_s_prime = ecu * (c - d_prime) / c
        fs_prime = min(fy, max(-fy, eps_s_prime * Es))
        Cs = As_prime * fs_prime
        if d_prime <= a and eps_s_prime > 0:
            Cs -= As_prime * 0.85 * fc
        eps_s = ecu * (d - c) / c
        fs = min(fy, max(-fy, eps_s * Es))
        T = As * fs
        if (Cc + Cs) <= T:
            converged = True
            break
        c -= 0.1

    Mn_kNm = (Cc * (d - a / 2) + Cs * (d - d_prime)) / 1_000_000
    eps_t = ecu * (dt - c) / c if c > 0 else 0
    if eps_t <= eps_y:
        phi = 0.65
        strain_class = "Compression-controlled"
    elif eps_t >= tension_control_limit:
        phi = 0.90
        strain_class = "Tension-controlled"
    else:
        phi = 0.65 + 0.25 * ((eps_t - eps_y) / 0.003)
        strain_class = "Transition zone"

    return {
        "phi_Mn": round(phi * Mn_kNm, 1),
        "is_ductile": eps_t >= tension_control_limit,
        "As_min": round(As_min, 1),
        "passes_As_min": As >= As_min,
        "As_max_tc": round(As_max_tc, 1),
        "passes_As_max_tc": As <= As_max_tc,
        "strain_class": strain_class,
        "converged": converged,
        "c": round(c, 2),
        "a": round(a, 2),
        "eps_t": round(eps_t, 5),
        "phi": round(phi, 3),
        "Mn": round(Mn_kNm, 1),
    }


def calculate_shear_torsion(b, h, d, fc, fyt, fyl, cover_clear, Vu_kN, Tu_kNm, n_legs, bar_dia, lambda_c):
    if d <= 0:
        return {"final_s": 0, "section_fails": True, "needs_torsion": False, "Al_req": 0, "Al_min": 0}

    phi_v = 0.75
    Vu = abs(Vu_kN) * 1000
    Tu = abs(Tu_kNm) * 1_000_000
    A_leg = math.pi * bar_dia**2 / 4
    lambda_s = min(1.0, math.sqrt(2 / (1 + 0.004 * d)))
    Vc = 0.17 * lambda_s * lambda_c * math.sqrt(fc) * b * d
    phi_Vc = phi_v * Vc

    x1 = max(1, b - 2 * (cover_clear + bar_dia / 2))
    y1 = max(1, h - 2 * (cover_clear + bar_dia / 2))
    Aoh = x1 * y1
    Ao = 0.85 * Aoh
    ph = 2 * (x1 + y1)
    Acp = b * h
    pcp = 2 * (b + h)
    T_th = phi_v * 0.083 * lambda_c * math.sqrt(fc) * (Acp**2 / pcp)
    needs_torsion = Tu > T_th

    Vs_req = max((Vu / phi_v) - Vc, 0) if Vu > phi_Vc / 2 else 0
    Av_s_req = Vs_req / (fyt * d) if Vs_req > 0 else 0

    if needs_torsion:
        At_s_req = (Tu / phi_v) / (2 * Ao * fyt)
        At_s_for_min = max(At_s_req, 0.175 * b / fyt)
        Al_min = max(0, (0.42 * math.sqrt(fc) * Acp / fyl) - (At_s_for_min * ph * (fyt / fyl)))
        Al_req = max(At_s_req * ph * (fyt / fyl), Al_min)
    else:
        At_s_req = Al_req = Al_min = 0

    req_per_outer_leg = (Av_s_req / max(n_legs, 1)) + At_s_req
    s_calc = A_leg / req_per_outer_leg if req_per_outer_leg > 0 else 9999
    min_combined_ratio = max(0.062 * math.sqrt(fc) * b / fyt, 0.35 * b / fyt)
    s_min_steel = A_leg / (min_combined_ratio / 2)
    s_req = min(s_calc, s_min_steel)
    s_max_shear = min(d / 4, 300) if Vs_req > (0.33 * math.sqrt(fc) * b * d) else min(d / 2, 600)
    s_max = min(s_max_shear, ph / 8, 300) if needs_torsion else s_max_shear
    s_exact = min(s_req, s_max)
    final_s = math.floor(s_exact / 25) * 25

    Vs_prov = (n_legs * A_leg * fyt * d / final_s) if final_s > 0 else 0
    phi_Vn = phi_v * (Vc + Vs_prov)
    v_stress = Vu / (b * d)
    t_stress = (Tu * ph) / (1.7 * Aoh**2) if needs_torsion else 0
    combined_stress = math.sqrt(v_stress**2 + t_stress**2)
    stress_limit = phi_v * ((Vc / (b * d)) + 0.66 * math.sqrt(fc))
    section_fails = combined_stress > stress_limit

    return {
        "final_s": 0 if final_s < 50 or section_fails else final_s,
        "section_fails": section_fails,
        "needs_torsion": needs_torsion,
        "Al_req": round(Al_req, 1),
        "Al_min": round(Al_min, 1),
        "T_th": round(T_th / 1_000_000, 1),
        "phi_Vc": round(phi_Vc / 1000, 1),
        "phi_Vn": round(phi_Vn / 1000, 1),
        "lambda_s": round(lambda_s, 3),
        "s_exact": round(s_exact, 1),
        "s_max": round(s_max, 1),
        "combined_stress": round(combined_stress, 2),
        "stress_limit": round(stress_limit, 2),
        "Aoh": Aoh,
        "Ao": Ao,
        "ph": ph,
    }


def calculate_development_length(db, fy, fc, is_top_bar, cover_clear, clear_spacing, lambda_c):
    if db == 0:
        return {"ld": 0, "lap": 0, "ldh": 0}
    psi_t = 1.3 if is_top_bar else 1.0
    psi_e = 1.0
    psi_s = 0.8 if db <= 20 else 1.0
    psi_g = 1.0 if fy <= 420 else 1.15
    cb = min(cover_clear + db / 2, clear_spacing / 2 + db / 2)
    conf_term = min(cb / db, 2.5)
    ld_calc = (fy / (1.1 * lambda_c * math.sqrt(fc))) * ((psi_t * psi_e * psi_s * psi_g) / conf_term) * db
    ld = max(ld_calc, 300)
    lap_splice = max(1.3 * ld, 300)
    ldh_calc = max((0.24 * psi_e * psi_g * fy / (lambda_c * math.sqrt(fc))) * db, 8 * db, 150)
    return {
        "ld": math.ceil(ld / 50) * 50,
        "lap": math.ceil(lap_splice / 50) * 50,
        "ldh": math.ceil(ldh_calc / 50) * 50,
    }


def status_card(kind, text):
    st.markdown(f"<div class='status-card status-{kind}'>{text}</div>", unsafe_allow_html=True)


def mini_metric(label, value, delta):
    st.markdown(
        f"""
<div class="mini-metric">
  <div class="mini-metric-label">{label}</div>
  <div class="mini-metric-value">{value}</div>
  <div class="mini-metric-delta">↗ {delta}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def check_row(label, ok, detail, warn=False):
    cls = "badge-warn" if warn else ("badge-pass" if ok else "badge-fail")
    txt = "WARN" if warn else ("PASS" if ok else "FAIL")
    st.markdown(
        f"""
<div class="check-row">
  <div class="check-label">{label}</div>
  <div class="check-detail">{detail}</div>
  <div class="badge {cls}">{txt}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def calculate_skin_reinforcement(h, d, skin_bar_dia, skin_bar_qty, skin_layers):
    required = h > 900
    s_limit = min(d / 6, 300) if d > 0 else 300
    zone_height = h / 2
    provided_layers = max(1, int(skin_layers))
    spacing = zone_height / (provided_layers - 1) if provided_layers > 1 else zone_height
    spacing_ok = (not required) or spacing <= s_limit
    bars_per_side = max(0, int(skin_bar_qty)) * provided_layers
    area_per_side = bars_per_side * math.pi * skin_bar_dia**2 / 4
    return {
        "required": required,
        "s_limit": round(s_limit, 1),
        "bars_per_side": bars_per_side,
        "bars_per_layer": max(0, int(skin_bar_qty)),
        "layers": provided_layers,
        "spacing": round(spacing, 1),
        "spacing_ok": spacing_ok,
        "area_per_side": round(area_per_side, 1),
        "zone_height": round(zone_height, 1),
    }


def draw_beam_section(b, h, cover, tie_dia, top_rg, bot_rg, flex, shear, zone, skin=None, skin_bar_dia=10):
    fig, ax = plt.subplots(figsize=(1.35, 1.15), dpi=160)
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.set_aspect("equal")
    ax.axis("off")

    pad = max(b, h) * 0.18
    ax.set_xlim(-pad, b + pad * 1.65)
    ax.set_ylim(h + pad * 0.78, -pad)

    ax.add_patch(patches.Rectangle((0, 0), b, h, facecolor="#1e2330", edgecolor="#4f8ef7", lw=1.2))
    ax.add_patch(
        patches.Rectangle(
            (cover + tie_dia / 2, cover + tie_dia / 2),
            b - 2 * (cover + tie_dia / 2),
            h - 2 * (cover + tie_dia / 2),
            fill=False,
            edgecolor="#c084fc",
            lw=1.1,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (cover, cover),
            b - 2 * cover,
            h - 2 * cover,
            fill=False,
            edgecolor="#7a84a0",
            lw=0.8,
        )
    )

    a = max(0, min(h, flex.get("a", 0)))
    c = max(0, min(h, flex.get("c", 0)))
    ax.add_patch(patches.Rectangle((0, 0), b, a, facecolor="#4f8ef7", alpha=0.20, edgecolor="none"))
    ax.plot([-pad * 0.18, b + pad * 0.18], [c, c], color="#f87171", lw=0.8, ls=(0, (4, 3)))
    ax.text(b + pad * 0.25, max(12, a / 2), f"a={flex.get('a', 0):.0f}", color="#93c5fd", fontsize=4.6)
    ax.text(b + pad * 0.25, c, f"c={flex.get('c', 0):.0f}", color="#f87171", fontsize=4.6, va="center")

    def plot_layers(group, from_top=True):
        for n, dia, y_from_face in group.layers:
            y = y_from_face if from_top else h - y_from_face
            usable = b - 2 * (cover + tie_dia + dia / 2)
            x0 = cover + tie_dia + dia / 2
            xs = [b / 2] if n == 1 else [x0 + i * usable / (n - 1) for i in range(n)]
            for x in xs:
                ax.add_patch(patches.Circle((x, y), max(dia / 2, 4), facecolor="#fbbf24", edgecolor="#92400e", lw=0.5))

    plot_layers(top_rg, True)
    plot_layers(bot_rg, False)

    corner_offset = cover + tie_dia + 10
    for x, y in [(corner_offset, corner_offset), (b - corner_offset, corner_offset), (corner_offset, h - corner_offset), (b - corner_offset, h - corner_offset)]:
        ax.add_patch(patches.Circle((x, y), 5, facecolor="#c084fc", edgecolor="#5b21b6", lw=0.5))

    if skin and skin["bars_per_side"] > 0:
        tension_top = zone in ["Left", "Right"]
        y_start = cover + tie_dia + skin_bar_dia / 2 if tension_top else h / 2
        y_end = h / 2 if tension_top else h - cover - tie_dia - skin_bar_dia / 2
        if skin["layers"] == 1:
            y_vals = [(y_start + y_end) / 2]
        else:
            y_vals = [y_start + i * (y_end - y_start) / (skin["layers"] - 1) for i in range(skin["layers"])]
        left_base = cover + tie_dia + skin_bar_dia / 2
        right_base = b - cover - tie_dia - skin_bar_dia / 2
        x_offsets = [(i - (skin["bars_per_layer"] - 1) / 2) * (skin_bar_dia * 0.75) for i in range(skin["bars_per_layer"])]
        for y in y_vals:
            for offset in x_offsets:
                for base, sign in [(left_base, 1), (right_base, -1)]:
                    x = base + sign * offset
                    ax.add_patch(patches.Circle((x, y), max(skin_bar_dia / 2, 3.5), facecolor="#38bdf8", edgecolor="#164e63", lw=0.45))

    ax.annotate("", xy=(0, h + pad * 0.25), xytext=(b, h + pad * 0.25), arrowprops=dict(arrowstyle="<->", color="#7a84a0", lw=0.55))
    ax.text(b / 2, h + pad * 0.34, f"b={b:.0f}", color="#e8eaf0", ha="center", fontsize=4.6)
    ax.annotate("", xy=(b + pad * 0.55, 0), xytext=(b + pad * 0.55, h), arrowprops=dict(arrowstyle="<->", color="#7a84a0", lw=0.55))
    ax.text(b + pad * 0.66, h / 2, f"h={h:.0f}", color="#e8eaf0", va="center", fontsize=4.8)
    ax.text(0, -pad * 0.18, f"{zone}", color="#98a2b8", fontsize=5.4, weight="bold")
    return fig


def draw_force_diagrams(forces, beam_length, df=None, selected_frame="Manual"):
    fig, (ax_m, ax_v) = plt.subplots(2, 1, figsize=(3.15, 1.55), dpi=180, sharex=True)
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax_m, ax_v):
        ax.set_facecolor("#181c24")
        ax.grid(True, color="#364060", alpha=0.36, linestyle="--", linewidth=0.28)
        ax.axhline(0, color="#e8eaf0", linewidth=0.45, alpha=0.75)
        ax.tick_params(colors="#98a2b8", labelsize=4.0, length=2, pad=1)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        for spine in ax.spines.values():
            spine.set_color("#2a3044")

    if df is not None and not df.empty and {"Station", "M3", "V2"}.issubset(df.columns):
        df_env = (
            df.groupby("Station")
            .agg(M3_Max=("M3", "max"), M3_Min=("M3", "min"), V2_Max=("V2", "max"), V2_Min=("V2", "min"))
            .reset_index()
            .sort_values("Station")
        )
        x = df_env["Station"]
        ax_m.plot(x, df_env["M3_Max"], color="#4f8ef7", linewidth=0.75, label="+M")
        ax_m.plot(x, df_env["M3_Min"], color="#f87171", linewidth=0.75, label="-M")
        ax_m.fill_between(x, df_env["M3_Min"], df_env["M3_Max"], color="#7a84a0", alpha=0.18)
        ax_v.plot(x, df_env["V2_Max"], color="#22c55e", linewidth=0.75, label="+V")
        ax_v.plot(x, df_env["V2_Min"], color="#fbbf24", linewidth=0.75, label="-V")
        ax_v.fill_between(x, df_env["V2_Min"], df_env["V2_Max"], color="#22c55e", alpha=0.12)
        title = f"Frame {selected_frame} - SAP2000 Envelope"
    else:
        L = max(float(beam_length or 0), 1.0)
        x = [0, 0.5 * L, L]
        m = [-abs(forces["Left"]["M"]), abs(forces["Mid"]["M"]), -abs(forces["Right"]["M"])]
        v = [abs(forces["Left"]["V"]), 0, -abs(forces["Right"]["V"])]
        ax_m.plot(x, m, color="#4f8ef7", linewidth=0.9, marker="o", markersize=2.2, label="M")
        ax_m.fill_between(x, m, 0, color="#4f8ef7", alpha=0.13)
        ax_v.plot(x, v, color="#22c55e", linewidth=0.9, marker="o", markersize=2.2, label="V")
        ax_v.fill_between(x, v, 0, color="#22c55e", alpha=0.13)
        ax_m.annotate(f"i -{abs(forces['Left']['M']):.0f}", (x[0], m[0]), color="#f87171", fontsize=4.8, xytext=(2, -7), textcoords="offset points")
        ax_m.annotate(f"mid +{abs(forces['Mid']['M']):.0f}", (x[1], m[1]), color="#4f8ef7", fontsize=4.8, xytext=(2, 4), textcoords="offset points")
        ax_m.annotate(f"j -{abs(forces['Right']['M']):.0f}", (x[2], m[2]), color="#f87171", fontsize=4.8, xytext=(-23, -7), textcoords="offset points")
        title = "Manual Design Demands"

    ax_m.set_title(title, color="#e8eaf0", fontsize=4.6, fontweight="bold", pad=2)
    ax_m.set_ylabel("M", color="#98a2b8", fontsize=4.2, labelpad=1)
    ax_v.set_ylabel("V", color="#98a2b8", fontsize=4.2, labelpad=1)
    ax_v.set_xlabel("Station (m)", color="#98a2b8", fontsize=4.2, labelpad=1)
    for ax in (ax_m, ax_v):
        ax.legend(
            facecolor="#181c24",
            edgecolor="#2a3044",
            labelcolor="#e8eaf0",
            fontsize=3.5,
            loc="upper left",
            borderpad=0.2,
            handlelength=1.6,
        )
    plt.tight_layout(pad=0.25, h_pad=0.25)
    return fig


def create_pdf_report(b, h, fc, fy, fyt, frame_name, zone_data, input_mode):
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 7, "RC Beam Design Calculation Package", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 5, f"Frame ID: {frame_name} | Code: ACI 318-19 | Input: {input_mode}", ln=True, align="C")
    pdf.cell(0, 5, f"Section: {b}x{h} mm | fc = {fc} MPa | fy = {fy} MPa | fyt = {fyt} MPa", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 6, "Zone Capacities and Detailing", ln=True)
    for zone in ["Left", "Mid", "Right"]:
        data = zone_data.get(zone)
        if not data:
            continue
        pdf.set_font("Arial", "B", 9)
        pdf.cell(17, 5, f"{zone}:", border=0)
        pdf.set_font("Arial", "", 9)
        pdf.cell(0, 5, f"Mu {data['Mu']} kNm | phiMn {data['phi_Mn']} kNm | Vu {data['Vu']} kN | {data['stirrups']}", ln=True)
        pdf.cell(17, 5, "", border=0)
        pdf.cell(0, 5, f"Top hook {data['dev_top']} mm | Top lap {data['dev_top_lap']} mm | Bottom lap {data['dev_bot']} mm", ln=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            pdf_bytes = f.read()
    try:
        os.remove(tmp_pdf.name)
    except OSError:
        pass
    return pdf_bytes


st.markdown(
    """
<div class="app-hero">
  <div class="app-hero-title">RC Beam Designer - ACI 318-19</div>
  <div class="app-hero-sub">Streamlit version adapted from your HTML/JavaScript UI: dark workspace, check cards, torsion indicators, and beam cross-section sketch.</div>
</div>
    """,
    unsafe_allow_html=True,
)

input_mode = st.radio("Force input source", ["Manual Input", "SAP2000 CSV Upload"], horizontal=True)
use_sap = input_mode == "SAP2000 CSV Upload"

forces = {"Left": {"M": 0.0, "V": 0.0, "T": 0.0}, "Mid": {"M": 0.0, "V": 0.0, "T": 0.0}, "Right": {"M": 0.0, "V": 0.0, "T": 0.0}}
beam_length = 6.0
selected_frame = "Manual"
df = None

if use_sap:
    uploaded_file = st.file_uploader("Upload SAP2000 frame-forces CSV", type=["csv"])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        if "Frame" in df_raw.columns and str(df_raw["Frame"].iloc[0]).strip().lower() == "text":
            df_raw = df_raw.drop(0).reset_index(drop=True)
        for col in ["Station", "P", "V2", "V3", "T", "M2", "M3"]:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
        if "Frame" not in df_raw.columns:
            st.error("CSV must contain a Frame column.")
        else:
            selected_frame = st.selectbox("Select beam frame", df_raw["Frame"].dropna().unique())
            df = df_raw[df_raw["Frame"] == selected_frame].copy()
            if "OutputCase" not in df.columns:
                df["OutputCase"] = "Manual"
            df["V2_abs"] = df["V2"].abs()
            df["T_abs"] = df["T"].abs()
            beam_length = float(df["Station"].max())
            df_left = df[df["Station"] <= 0.1 * beam_length]
            df_right = df[df["Station"] >= 0.9 * beam_length]
            df_mid = df[(df["Station"] > 0.3 * beam_length) & (df["Station"] < 0.7 * beam_length)]
            forces["Left"]["M"] = df_left["M3"].abs().max() if not df_left.empty else 0
            forces["Left"]["V"] = df_left["V2_abs"].max() if not df_left.empty else 0
            forces["Left"]["T"] = df_left["T_abs"].max() if not df_left.empty else 0
            forces["Right"]["M"] = df_right["M3"].abs().max() if not df_right.empty else 0
            forces["Right"]["V"] = df_right["V2_abs"].max() if not df_right.empty else 0
            forces["Right"]["T"] = df_right["T_abs"].max() if not df_right.empty else 0
            forces["Mid"]["M"] = df_mid["M3"].abs().max() if not df_mid.empty else df["M3"].abs().max()
            forces["Mid"]["V"] = df_mid["V2_abs"].max() if not df_mid.empty else 0
            forces["Mid"]["T"] = df_mid["T_abs"].max() if not df_mid.empty else 0
            st.info(f"Beam {selected_frame}, L = {beam_length:.2f} m. ACI deflection h_min note: about {beam_length * 1000 / 18.5:.1f} mm for one-end-continuous.")
    else:
        st.info("Upload a SAP2000 CSV file to begin.")
else:
    st.markdown("<div class='section-band'>Manual Force Input</div>", unsafe_allow_html=True)
    beam_length = st.number_input("Beam span L (m)", value=6.0, step=0.5, min_value=1.0)
    st.info(f"ACI deflection h_min note: about {beam_length * 1000 / 18.5:.1f} mm for one-end-continuous. Adjust for actual end conditions.")
    c1, c2, c3 = st.columns(3)
    defaults = {"Left": (200.0, 150.0, 0.0), "Mid": (180.0, 60.0, 0.0), "Right": (220.0, 155.0, 0.0)}
    for col, zone in zip([c1, c2, c3], ["Left", "Mid", "Right"]):
        with col:
            st.subheader(zone)
            forces[zone]["M"] = st.number_input(f"Mu {zone} (kNm)", value=defaults[zone][0], step=5.0, min_value=0.0)
            forces[zone]["V"] = st.number_input(f"Vu {zone} (kN)", value=defaults[zone][1], step=5.0, min_value=0.0)
            forces[zone]["T"] = st.number_input(f"Tu {zone} (kNm)", value=defaults[zone][2], step=1.0, min_value=0.0)

st.markdown("<div class='section-band'>Bending and Shear Diagram</div>", unsafe_allow_html=True)
force_fig = draw_force_diagrams(forces, beam_length, df=df, selected_frame=selected_frame)
st.pyplot(force_fig, use_container_width=False)
plt.close(force_fig)

st.markdown("<div class='section-band'>Project Input Workspace</div>", unsafe_allow_html=True)
col_prop, col_rebar = st.columns([1, 2])

with col_prop:
    st.subheader("Section and Materials")
    b = st.number_input("Width b (mm)", value=300, step=50, min_value=150)
    h = st.number_input("Total depth h (mm)", value=600, step=50, min_value=200)
    fc = st.number_input("Concrete fc' (MPa)", value=35, step=5, min_value=20)
    fy = st.number_input("Main steel fy (MPa)", value=500, step=10, min_value=300)
    lambda_c = st.selectbox("Concrete type lambda", [1.0, 0.85, 0.75], format_func=lambda x: {1.0: "1.0 Normal weight", 0.85: "0.85 Sand-lightweight", 0.75: "0.75 All-lightweight"}[x])
    st.subheader("Transverse Steel")
    fyt = st.number_input("Stirrup fy (MPa)", value=400, step=10, min_value=240)
    bar_v_options = {"RB9": 9, "DB10": 10, "DB12": 12, "DB16": 16}
    bar_v_name = st.selectbox("Stirrup size", list(bar_v_options.keys()), index=1)
    bar_v = bar_v_options[bar_v_name]
    n_legs = st.number_input("Stirrup legs", min_value=2, value=2, step=1)
    cover_clear = st.number_input("Clear cover to stirrup (mm)", value=40, step=5, min_value=20)
    clear_space = st.number_input("Minimum clear bar spacing (mm)", value=25, step=5, min_value=20)
    st.subheader("Skin Bars")
    skin_bar_options = {"DB10": 10, "DB12": 12, "DB16": 16, "DB20": 20}
    skin_c1, skin_c2, skin_c3 = st.columns(3)
    skin_bar_qty = skin_c1.number_input(
        "Qty/layer each side",
        min_value=0,
        value=2,
        step=1,
        help="Example: 2 with DB12 means 2-DB12 at each skin-bar layer on each side face.",
    )
    skin_bar_name = skin_c2.selectbox(
        "Skin bar size",
        list(skin_bar_options.keys()),
        index=1,
        help="ACI side-face longitudinal reinforcement is checked when overall beam depth exceeds 900 mm.",
    )
    skin_layers = skin_c3.number_input(
        "Skin layers",
        min_value=1,
        value=2,
        step=1,
        help="Example: 2 layers of 2-DB12 side-face bars.",
    )
    skin_bar_dia = skin_bar_options[skin_bar_name]

with col_rebar:
    st.subheader("Zone Reinforcement")
    bar_opts = {"DB12": 12, "DB16": 16, "DB20": 20, "DB25": 25, "DB28": 28, "DB32": 32}
    tabs = st.tabs(["Left Support", "Midspan", "Right Support"])
    rebar_data = {}
    bar_selections = {}
    for i, zone in enumerate(["Left", "Mid", "Right"]):
        with tabs[i]:
            def_t = 4 if zone in ["Left", "Right"] else 2
            def_b = 4 if zone == "Mid" else 2
            top_col, bot_col = st.columns(2)
            with top_col:
                st.markdown("**Top bars**")
                t_n1 = st.number_input("Top L1 n", 0, value=def_t, key=f"t1_{zone}")
                t_d1_name = st.selectbox("Top L1 size", list(bar_opts.keys()), index=3, key=f"td1_{zone}")
                t_n2 = st.number_input("Top L2 n", 0, value=0, key=f"t2_{zone}")
                t_d2_name = st.selectbox("Top L2 size", list(bar_opts.keys()), index=2, key=f"td2_{zone}")
                t_n3 = st.number_input("Top L3 n", 0, value=0, key=f"t3_{zone}")
                t_d3_name = st.selectbox("Top L3 size", list(bar_opts.keys()), index=2, key=f"td3_{zone}")
            with bot_col:
                st.markdown("**Bottom bars**")
                b_n1 = st.number_input("Bottom L1 n", 0, value=def_b, key=f"b1_{zone}")
                b_d1_name = st.selectbox("Bottom L1 size", list(bar_opts.keys()), index=3, key=f"bd1_{zone}")
                b_n2 = st.number_input("Bottom L2 n", 0, value=0, key=f"b2_{zone}")
                b_d2_name = st.selectbox("Bottom L2 size", list(bar_opts.keys()), index=2, key=f"bd2_{zone}")
                b_n3 = st.number_input("Bottom L3 n", 0, value=0, key=f"b3_{zone}")
                b_d3_name = st.selectbox("Bottom L3 size", list(bar_opts.keys()), index=2, key=f"bd3_{zone}")
            t_d1, t_d2, t_d3 = bar_opts[t_d1_name], bar_opts[t_d2_name], bar_opts[t_d3_name]
            b_d1, b_d2, b_d3 = bar_opts[b_d1_name], bar_opts[b_d2_name], bar_opts[b_d3_name]
            rebar_data[zone] = {
                "top": get_rebar_group(t_n1, t_d1, t_n2, t_d2, t_n3, t_d3, cover_clear, bar_v, clear_space),
                "bot": get_rebar_group(b_n1, b_d1, b_n2, b_d2, b_n3, b_d3, cover_clear, bar_v, clear_space),
            }
            bar_selections[zone] = {"top_d1": t_d1 if t_n1 > 0 else 0, "bot_d1": b_d1 if b_n1 > 0 else 0}

st.markdown("<div class='section-band'>Run Design</div>", unsafe_allow_html=True)
if st.button("Run full 3-zone detailing design", type="primary", use_container_width=True):
    st.session_state["design_results_visible"] = True

if st.session_state.get("design_results_visible", False):
    pdf_zone_data = {}
    summary_rows = []
    st.markdown("<div class='section-band'>Three-Zone Cross Sections and Calculations</div>", unsafe_allow_html=True)
    result_columns = st.columns(3, gap="small")

    for idx, zone in enumerate(["Left", "Mid", "Right"]):
        with result_columns[idx]:
            display_title = {"Left": "Left Support", "Mid": "Mid", "Right": "Right Support"}[zone]
            zone_label = {"Left": "i - left support", "Mid": "midspan", "Right": "j - right support"}[zone]
            st.markdown(f"<div class='zone-title'>{display_title}</div>", unsafe_allow_html=True)
            st.caption(f"{zone} Section ({zone_label})")
            top_rg = rebar_data[zone]["top"]
            bot_rg = rebar_data[zone]["bot"]
            if top_rg.width_req > b or bot_rg.width_req > b:
                status_card("fail", f"Bars do not fit in {b} mm width. Top requires {top_rg.width_req:.1f} mm, bottom requires {bot_rg.width_req:.1f} mm.")
                pdf_zone_data[zone] = None
                continue

            if zone == "Mid":
                As_tens, As_comp = bot_rg.area, top_rg.area
                d = h - bot_rg.centroid
                dt = h - bot_rg.extreme_fiber
                d_prime = top_rg.centroid
            else:
                As_tens, As_comp = top_rg.area, bot_rg.area
                d = h - top_rg.centroid
                dt = h - top_rg.extreme_fiber
                d_prime = bot_rg.centroid

            Mu = abs(forces[zone]["M"])
            Vu_design = abs(forces[zone]["V"])
            res_flex = calculate_beam_flexure(b, h, d, dt, d_prime, fc, fy, As_tens, As_comp)
            res_shear = calculate_shear_torsion(b, h, d, fc, fyt, fy, cover_clear, Vu_design, forces[zone]["T"], n_legs, bar_v, lambda_c)
            skin = calculate_skin_reinforcement(h, d, skin_bar_dia, skin_bar_qty, skin_layers)
            dev_top = calculate_development_length(bar_selections[zone]["top_d1"], fy, fc, True, cover_clear, clear_space, lambda_c)
            dev_bot = calculate_development_length(bar_selections[zone]["bot_d1"], fy, fc, False, cover_clear, clear_space, lambda_c)
            dc_flex = round(Mu / res_flex["phi_Mn"], 2) if res_flex["phi_Mn"] > 0 else 999.9
            dc_shear = round(Vu_design / res_shear["phi_Vn"], 2) if res_shear.get("phi_Vn", 0) > 0 else 999.9

            flex_ok = res_flex["converged"] and res_flex["passes_As_min"] and res_flex["is_ductile"] and res_flex["phi_Mn"] >= Mu
            shear_ok = res_shear["final_s"] > 0
            if flex_ok and shear_ok:
                status_card("pass", f"{zone} passes flexure and shear/torsion preliminary checks.")
            elif res_flex["phi_Mn"] >= Mu and shear_ok:
                status_card("warn", f"{zone} has enough strength, but detailing or ductility needs review.")
            else:
                status_card("fail", f"{zone} fails one or more required checks.")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                mini_metric("phi Mn", f"{res_flex['phi_Mn']} kNm", f"D/C {dc_flex}")
            with m2:
                mini_metric("phi Vn", f"{res_shear['phi_Vn']} kN", f"D/C {dc_shear}")
            with m3:
                mini_metric("Stirrups", f"{n_legs}-{bar_v_name}", f"@ {res_shear['final_s']} mm" if res_shear["final_s"] else "FAIL")
            with m4:
                mini_metric("Strain", f"{res_flex['eps_t']}", res_flex["strain_class"])

            fig = draw_beam_section(b, h, cover_clear, bar_v, top_rg, bot_rg, res_flex, res_shear, zone, skin, skin_bar_dia)
            img_pad_l, img_mid, img_pad_r = st.columns([0.22, 0.56, 0.22])
            with img_mid:
                st.pyplot(fig, use_container_width=False)
            plt.close(fig)

            st.markdown("<div class='section-band'>ACI Style Checks</div>", unsafe_allow_html=True)
            check_row("Flexure phiMn >= Mu", res_flex["phi_Mn"] >= Mu, f"{res_flex['phi_Mn']} >= {Mu:.1f} kNm")
            check_row("Minimum As", res_flex["passes_As_min"], f"As = {As_tens:.1f}; As,min = {res_flex['As_min']} mm2")
            check_row("Tension-controlled", res_flex["is_ductile"], f"eps_t = {res_flex['eps_t']}; phi = {res_flex['phi']}", warn=not res_flex["is_ductile"] and res_flex["phi_Mn"] >= Mu)
            check_row("Shear phiVn >= Vu", res_shear["phi_Vn"] >= Vu_design, f"{res_shear['phi_Vn']} >= {Vu_design:.1f} kN")
            check_row("Transverse spacing", res_shear["final_s"] > 0, f"s exact = {res_shear['s_exact']} mm; s max = {res_shear['s_max']} mm")
            check_row("Torsion threshold", not res_shear["needs_torsion"], f"Tu = {forces[zone]['T']:.1f} kNm; phiTth = {res_shear['T_th']} kNm", warn=res_shear["needs_torsion"])
            skin_detail = (
                f"{skin['layers']} layers of {skin['bars_per_layer']}-{skin_bar_name} each side; h = {h:.0f} mm <= 900 mm, not required"
                if not skin["required"]
                else f"{skin['layers']} layers of {skin['bars_per_layer']}-{skin_bar_name} each side; s = {skin['spacing']} <= {skin['s_limit']} mm"
            )
            check_row("ACI side-face skin bars", skin["spacing_ok"], skin_detail)

            if st.toggle(f"Show calculation summary - {zone}", value=False, key=f"calc_summary_{zone}"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            ("d", f"{d:.1f} mm", "Effective depth"),
                            ("d'", f"{d_prime:.1f} mm", "Compression steel depth"),
                            ("c / a", f"{res_flex['c']} / {res_flex['a']} mm", "Neutral axis and stress block"),
                            ("Mn / phiMn", f"{res_flex['Mn']} / {res_flex['phi_Mn']} kNm", "Nominal and design moment"),
                            ("lambda_s", res_shear["lambda_s"], "ACI size effect factor"),
                            ("Aoh / ph", f"{res_shear['Aoh']:.0f} mm2 / {res_shear['ph']:.0f} mm", "Torsion cage geometry"),
                            ("Al torsion", f"{res_shear['Al_req']} mm2", "Required if torsion governs"),
                            ("Skin bars", skin_detail, "ACI 318 side-face longitudinal reinforcement for h > 900 mm"),
                            ("Top ldh / lap", f"{dev_top['ldh']} / {dev_top['lap']} mm", "Development lengths"),
                            ("Bottom lap", f"{dev_bot['lap']} mm", "Development length"),
                        ],
                        columns=["Parameter", "Value", "Note"],
                    ),
                    hide_index=True,
                )

            stirrup_text = f"{n_legs}-DB{bar_v} @ {res_shear['final_s']} mm (D/C: {dc_shear})" if res_shear["final_s"] > 0 else "FAILS"
            pdf_zone_data[zone] = {
                "Mu": round(Mu, 1),
                "phi_Mn": res_flex["phi_Mn"],
                "DC_flex": dc_flex,
                "Vu": round(Vu_design, 1),
                "DC_shear": dc_shear,
                "stirrups": stirrup_text,
                "dev_top": dev_top["ldh"],
                "dev_top_lap": dev_top["lap"],
                "dev_bot": dev_bot["lap"],
            }
            summary_rows.append(
                {
                    "Zone": zone,
                    "Mu (kNm)": round(Mu, 1),
                    "phiMn (kNm)": res_flex["phi_Mn"],
                    "Flexure D/C": dc_flex,
                    "Vu (kN)": round(Vu_design, 1),
                    "phiVn (kN)": res_shear["phi_Vn"],
                    "Stirrups": stirrup_text,
                    "Strain class": res_flex["strain_class"],
                }
            )

    if summary_rows:
        st.markdown("<div class='section-band'>Design Summary Table</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        pdf_bytes = create_pdf_report(b, h, fc, fy, fyt, selected_frame, pdf_zone_data, input_mode)
        st.download_button("Download PDF calculation report", data=pdf_bytes, file_name=f"Beam_{selected_frame}_Report.pdf", mime="application/pdf", type="primary")

st.markdown(
    """
<div class="notice">
  <strong>Scope note:</strong> This adapts the HTML/JavaScript interface into Streamlit and keeps the calculation package preliminary.
  Torsion web-crushing interaction, hook detailing, seismic detailing, development/anchorage conditions, and final bar layout should be verified against the official ACI standard before issuing drawings.
</div>
    """,
    unsafe_allow_html=True,
)
