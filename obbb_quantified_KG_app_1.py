# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import altair as alt
import gdown

# =============== CONFIG ===============
# Google Drive FILE IDs (share files "Anyone with the link (Viewer)")
DF_JOINED_FILE_ID = "1vJfS10pDpAldKd1MhHptQZhrq5IcQxu_"   # e.g., '1AbC...'
PREDS_FILE_ID     = "1gbgrifQ0tMoKTIzAENnG9C9Vi-FyW7e8"       # e.g., '1XyZ...'

DATA_DIR = Path("data")   # local cache folder for downloaded files
DATA_DIR.mkdir(exist_ok=True)

TARGET_COL = "DISCHARGES"
EXOG_COLS = [
    "median_household_income","population","median_age","per_capita_income",
    "poverty_count","poverty_universe","medicaid_ins_total","medicare_ins_total",
    "priv_ins_total","no_ins_total"
]

# OBBB dials
RTR_MIN, RTR_MAX, RTR_STEP, RTR_DEFAULT = -100_000_000, 100_000_000, 1_000_000, 0

# =============== PAGE SETUP ===============
st.set_page_config(page_title="CALIBER360 OBBB Quantification", layout="wide", initial_sidebar_state="expanded")
st.title("CALIBER360 OBBB Quantification")
st.markdown("**OBBB impact, quantified for every California hospital—by ZIP, Service Line, and Payer Mix.**")
st.caption(
    "Discharges include **Ambulatory Surgery (AS) Only**, **Inpatient Only**, "
    "**Emergency Department (ED) Only**, and **Inpatient from ED**."
)

# Scrollbars for long dropdown menus
st.markdown("""
<style>
div[data-baseweb="menu"] { max-height: 320px !important; overflow-y: auto !important; }
</style>
""", unsafe_allow_html=True)

# =============== HELPERS ===============
def _normalize_zip(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)
    return s

def _int_nonneg(series: pd.Series) -> pd.Series:
    arr = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy(dtype=float)
    arr = np.maximum(arr, 0.0)
    return pd.Series(np.rint(arr).astype(int), index=series.index)

def _fmt_pct_signed(v: int) -> str:
    return f"{v:+d}%"

def _fmt_money_signed(v: int | float) -> str:
    sign = "+" if v > 0 else ("-" if v < 0 else "")
    return f"{sign}${abs(v):,}"

def _ensure_drive_file(file_id: str, local_path: Path) -> Path:
    """Download from Google Drive (if missing) into local cache."""
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    url = f"https://drive.google.com/uc?id={file_id}"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(local_path), quiet=True)
    return local_path

@st.cache_data(show_spinner=False, ttl=24*3600)
def load_df_joined_from_drive(file_id: str) -> pd.DataFrame:
    local_path = DATA_DIR / "df_joined.csv.gz"
    _ensure_drive_file(file_id, local_path)
    df = pd.read_csv(local_path, compression="gzip", low_memory=False, dtype={"PZIP": "string"})
    df.columns = df.columns.str.strip()

    req = {"FACILITY_NAME","PZIP","YEAR", TARGET_COL}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"`df_joined` missing columns: {sorted(missing)}")

    df["PZIP"] = _normalize_zip(df["PZIP"])
    # drop invalid zips / '00000'
    df = df[df["PZIP"].str.fullmatch(r"\d{5}") & (df["PZIP"] != "00000")].copy()

    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    for c in [TARGET_COL] + [c for c in EXOG_COLS if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    NA_SENTINELS = [-666666666, -66666666, -999999999, -99999999]
    present_exogs = [c for c in EXOG_COLS if c in df.columns]
    if present_exogs:
        df[present_exogs] = df[present_exogs].replace(NA_SENTINELS, np.nan)

    df = df.dropna(subset=["YEAR"]).copy()
    df["YEAR"] = df["YEAR"].astype(int)
    df = (df.sort_values(["FACILITY_NAME","PZIP","YEAR"])
            .drop_duplicates(["FACILITY_NAME","PZIP","YEAR"], keep="last")
            .reset_index(drop=True))
    return df

@st.cache_data(show_spinner=False, ttl=24*3600)
def load_preds_from_drive(file_id: str) -> pd.DataFrame:
    local_path = DATA_DIR / "preds.csv.gz"
    _ensure_drive_file(file_id, local_path)
    df = pd.read_csv(local_path, compression="gzip", low_memory=False, dtype={"PZIP": "string"})
    df.columns = df.columns.str.strip()

    req = {"FACILITY_NAME","PZIP","YEAR","PRED_DISCHARGES"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"`preds` missing columns: {sorted(missing)}")

    df["PZIP"] = _normalize_zip(df["PZIP"])
    df = df[df["PZIP"].str.fullmatch(r"\d{5}") & (df["PZIP"] != "00000")].copy()

    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    df["PRED_DISCHARGES"] = pd.to_numeric(df["PRED_DISCHARGES"], errors="coerce")

    df = df.dropna(subset=["YEAR"]).copy()
    df["YEAR"] = df["YEAR"].astype(int)
    df = (df.sort_values(["FACILITY_NAME","PZIP","YEAR"])
            .drop_duplicates(["FACILITY_NAME","PZIP","YEAR"], keep="last")
            .reset_index(drop=True))
    return df

def build_series(df_joined: pd.DataFrame, preds: pd.DataFrame, facility: str, selected_pzips: list[str]):
    # Actuals (2020–2024; 2024 may be missing)
    hist = (df_joined.query("FACILITY_NAME == @facility and 2020 <= YEAR <= 2024 and PZIP in @selected_pzips")
                    .groupby("YEAR", as_index=False)[TARGET_COL].sum()
                    .rename(columns={TARGET_COL: "VALUE"})
                    .sort_values("YEAR"))
    hist["VALUE"] = _int_nonneg(hist["VALUE"])

    # Forecasts (2024–2026)
    fc = (preds.query("FACILITY_NAME == @facility and 2024 <= YEAR <= 2026 and PZIP in @selected_pzips")
                .groupby("YEAR", as_index=False)["PRED_DISCHARGES"].sum()
                .rename(columns={"PRED_DISCHARGES": "VALUE"})
                .sort_values("YEAR"))
    fc["VALUE"] = _int_nonneg(fc["VALUE"])

    # Ensure 2024 appears as anchor on the solid line
    if 2024 not in hist["YEAR"].tolist():
        fc_2024 = fc.loc[fc["YEAR"] == 2024]
        hist_2020_2024 = pd.concat([hist, fc_2024], ignore_index=True).sort_values("YEAR") if not fc_2024.empty else hist.copy()
    else:
        hist_2020_2024 = hist.copy()

    fc_2025_2026 = fc.loc[fc["YEAR"] >= 2025].copy()
    fc_2025_2026["VALUE"] = _int_nonneg(fc_2025_2026["VALUE"])

    combined = pd.concat(
        [hist_2020_2024.assign(KIND="Actual (2020–2024)"),
         fc_2025_2026.assign(KIND="Forecast (2025–2026)")],
        ignore_index=True
    ).sort_values("YEAR")
    return hist_2020_2024, fc_2025_2026, combined

def plot_series(facility: str, hist_2020_2024: pd.DataFrame, fc_2025_2026: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9,5))
    if not hist_2020_2024.empty:
        ax.plot(hist_2020_2024["YEAR"], hist_2020_2024["VALUE"], marker="o", linestyle="-",
                color="tab:blue", label="Actual (2020–2024)")
    if not fc_2025_2026.empty:
        fc_line = fc_2025_2026
        if not hist_2020_2024.empty:
            anchor = hist_2020_2024.tail(1)[["YEAR","VALUE"]]
            fc_line = pd.concat([anchor, fc_2025_2026], ignore_index=True)
        ax.plot(fc_line["YEAR"], fc_line["VALUE"], marker="o", linestyle="--",
                color="tab:orange", label="Forecast (2025–2026)")
    ax.set_xticks(list(range(2020, 2027)))
    ax.set_xlabel("Year"); ax.set_ylabel("Discharges (total)")
    ax.set_title(f"{facility}: Discharges 2020–2026")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

def _sum_int_nonneg(s):
    return int(np.maximum(pd.to_numeric(s, errors="coerce").fillna(0), 0).sum().round())

@st.cache_data(show_spinner=False)
def zip_metrics_for_facility(df_joined, preds, facility: str, baseline_year=2024, forecast_year=2025):
    # base (prefer actual 2024; fallback to predicted 2024)
    base_actual = (df_joined.query("FACILITY_NAME == @facility and YEAR == @baseline_year")
                           .groupby("PZIP", as_index=False)[TARGET_COL].sum()
                           .rename(columns={TARGET_COL: "BASE"}))
    base_pred = (preds.query("FACILITY_NAME == @facility and YEAR == @baseline_year")
                        .groupby("PZIP", as_index=False)["PRED_DISCHARGES"].sum()
                        .rename(columns={"PRED_DISCHARGES": "BASE"}))
    base = pd.merge(base_pred, base_actual, on="PZIP", how="outer", suffixes=("_pred",""))
    base["BASE"] = base["BASE"].fillna(base["BASE_pred"]); base = base[["PZIP","BASE"]].fillna(0)

    fc = (preds.query("FACILITY_NAME == @facility and YEAR == @forecast_year")
                .groupby("PZIP", as_index=False)["PRED_DISCHARGES"].sum()
                .rename(columns={"PRED_DISCHARGES": "FC"}))

    out = base.merge(fc, on="PZIP", how="outer").fillna(0)
    out["BASE"] = out["BASE"].clip(lower=0).round().astype(int)
    out["FC"]   = out["FC"].clip(lower=0).round().astype(int)
    out["DELTA"] = out["FC"] - out["BASE"]
    out["PCT_DELTA"] = np.where(out["BASE"]>0, (out["DELTA"]/out["BASE"])*100.0, np.nan)
    return out.sort_values("DELTA", ascending=False).reset_index(drop=True)

def evidence_rows(df_joined, preds, facility: str, year: int, pzip: str|None):
    act = df_joined.query("FACILITY_NAME == @facility and YEAR == @year")
    if pzip and pzip != "All ZIPs": act = act.query("PZIP == @pzip")
    act = act.assign(SOURCE="actual", VALUE=act[TARGET_COL])[["FACILITY_NAME","PZIP","YEAR","VALUE","SOURCE"]]

    fc = preds.query("FACILITY_NAME == @facility and YEAR == @year")
    if pzip and pzip != "All ZIPs": fc = fc.query("PZIP == @pzip")
    fc = fc.assign(SOURCE="forecast", VALUE=fc["PRED_DISCHARGES"])[["FACILITY_NAME","PZIP","YEAR","VALUE","SOURCE"]]

    out = pd.concat([act, fc], ignore_index=True)
    out["VALUE"] = np.maximum(pd.to_numeric(out["VALUE"], errors="coerce").fillna(0), 0).round().astype(int)
    return out.sort_values(["SOURCE","PZIP"])

# =============== LOAD DATA (with custom spinner) ===============
with st.spinner("CALIBER360 loading data for all CA hospitals — please hold on..."):
    try:
        df_joined = load_df_joined_from_drive(DF_JOINED_FILE_ID)
        preds     = load_preds_from_drive(PREDS_FILE_ID)
    except Exception as e:
        st.error(f"Data load failed: {e}")
        st.stop()

# =============== SIDEBAR OBBB IMPACT ===============
with st.sidebar:
    st.subheader("OBBB Impact")
    medicaid_delta  = st.slider("Medicaid impact (%)",  -20, 20, 0, step=1)
    uninsured_delta = st.slider("Uninsured impact (%)", -20, 20, 0, step=1)
    medicare_delta  = st.slider("Medicare impact (%)",  -20, 20, 0, step=1)
    rtr_amount = st.slider("Rural Transformation Funds (USD)",
                           min_value=RTR_MIN, max_value=RTR_MAX, value=RTR_DEFAULT, step=RTR_STEP)
    st.caption(
        "Current OBBB impact settings:  "
        f"Medicaid {_fmt_pct_signed(medicaid_delta)} · "
        f"Uninsured {_fmt_pct_signed(uninsured_delta)} · "
        f"Medicare {_fmt_pct_signed(medicare_delta)} · "
        f"Rural Funds {_fmt_money_signed(rtr_amount)}.  "
        "_Not applied to this chart; available in revenue model._"
    )
    st.session_state.update({
        "knob_medicaid_pct": medicaid_delta,
        "knob_uninsured_pct": uninsured_delta,
        "knob_medicare_pct": medicare_delta,
        "knob_rural_relief_usd": rtr_amount,
    })

# =============== MAIN CONTROLS ===============
facilities_all = sorted(set(df_joined["FACILITY_NAME"].unique()).union(set(preds["FACILITY_NAME"].unique())))
facility = st.selectbox("Facility", facilities_all, index=0, help="Type to search, or scroll the list.")

pzips_for_fac = sorted(set(
    df_joined.loc[df_joined["FACILITY_NAME"] == facility, "PZIP"].unique()
).union(
    preds.loc[preds["FACILITY_NAME"] == facility, "PZIP"].unique()
))
# belt & suspenders: ensure valid and no 00000
pzips_for_fac = [z for z in pzips_for_fac if isinstance(z, str) and len(z)==5 and z.isdigit() and z!="00000"]

zip_options = ["All ZIPs"] + pzips_for_fac
zip_choice = st.selectbox("ZIP code (PZIP)", options=zip_options, index=0, help="Type to search, or scroll the list.")
selected_pzips = pzips_for_fac if zip_choice == "All ZIPs" else [zip_choice]

# =============== MAIN CHART ===============
if not pzips_for_fac:
    st.warning("No ZIPs found for this facility in the loaded data.")
else:
    hist_2020_2024, fc_2025_2026, combined = build_series(df_joined, preds, facility, selected_pzips)
    if hist_2020_2024.empty and fc_2025_2026.empty:
        st.warning("No data available for the selected filters.")
    else:
        plot_series(facility, hist_2020_2024, fc_2025_2026)

# =============== INSIGHT TABS ===============
tabs = st.tabs(["ZIP Bar Heat", "Recommendations", "Why?"])

with tabs[0]:
    st.subheader("ZIP Bar Heat — where growth concentrates")
    col1, col2 = st.columns([1,1])
    with col1:
        heat_year = st.selectbox("Forecast year", [2025, 2026], index=0, key="heat_year")
    with col2:
        color_metric = st.selectbox("Metric", ["Δ vs 2024 (count)", "%Δ vs 2024"], index=0, key="heat_metric")

    mdf = zip_metrics_for_facility(df_joined, preds, facility, baseline_year=2024, forecast_year=heat_year)

    show_col = "DELTA" if color_metric.startswith("Δ") else "PCT_DELTA"
    y_title  = "Δ (count)" if show_col == "DELTA" else "%Δ"

    # Clean up NaNs/Infs for sorting
    mdf = mdf.replace([np.inf, -np.inf], np.nan).dropna(subset=[show_col])

    # Top 10 increases (strictly > 0) and top 10 decreases (strictly < 0)
    inc = mdf[mdf[show_col] > 0].sort_values(show_col, ascending=False).head(10)
    dec = mdf[mdf[show_col] < 0].sort_values(show_col, ascending=True).head(10)

    c1, c2 = st.columns(2)

    if inc.empty and dec.empty:
        st.info("No increases or decreases found for the selected facility/year.")
    else:
        with c1:
            st.markdown("**Top 10 increases**")
            if inc.empty:
                st.caption("No positive changes.")
            else:
                chart_inc = (
                    alt.Chart(inc)
                       .mark_bar()
                       .encode(
                           x=alt.X("PZIP:N", sort='-y', title="ZIP"),
                           y=alt.Y(f"{show_col}:Q", title=y_title),
                           color=alt.value("#2e7d32"),  # green
                           tooltip=["PZIP","BASE","FC","DELTA","PCT_DELTA"]
                       )
                       .properties(height=360)
                )
                st.altair_chart(chart_inc, use_container_width=True)

        with c2:
            st.markdown("**Top 10 decreases**")
            if dec.empty:
                st.caption("No negative changes.")
            else:
                chart_dec = (
                    alt.Chart(dec)
                       .mark_bar()
                       .encode(
                           x=alt.X("PZIP:N", sort='y', title="ZIP"),
                           y=alt.Y(f"{show_col}:Q", title=y_title),
                           color=alt.value("#c62828"),  # red
                           tooltip=["PZIP","BASE","FC","DELTA","PCT_DELTA"]
                       )
                       .properties(height=360)
                )
                st.altair_chart(chart_dec, use_container_width=True)


with tabs[1]:
    st.subheader("Recommendations — focus ZIPs")
    rcol1, rcol2, rcol3 = st.columns([1,1,1])
    with rcol1:
        rec_year = st.selectbox("Forecast year", [2025, 2026], index=0, key="rec_year")
    with rcol2:
        top_k = st.slider("How many ZIPs?", 5, 50, 10, step=5)
    with rcol3:
        sort_by = st.selectbox("Rank by", ["Δ vs 2024 (count)", "%Δ vs 2024", "Forecast volume"], index=0)

    rdf = zip_metrics_for_facility(df_joined, preds, facility, baseline_year=2024, forecast_year=rec_year)
    if sort_by == "Forecast volume":
        rdf = rdf.sort_values("FC", ascending=False)
    elif sort_by.startswith("%Δ"):
        rdf = rdf.sort_values("PCT_DELTA", ascending=False)
    else:
        rdf = rdf.sort_values("DELTA", ascending=False)

    st.dataframe(rdf.head(top_k).rename(columns={
        "PZIP":"ZIP", "BASE":"Base 2024", "FC":f"Forecast {rec_year}",
        "DELTA":"Δ", "PCT_DELTA":"%Δ"
    }), use_container_width=True)

with tabs[2]:
    st.subheader("Why? — evidence for a given point")
    wcol1, wcol2 = st.columns([1,1])
    with wcol1:
        why_year = st.selectbox("Year", [2020,2021,2022,2023,2024,2025,2026], index=4, key="why_year")
    with wcol2:
        why_zip = st.selectbox("ZIP (or all)", ["All ZIPs"] + pzips_for_fac, index=0, key="why_zip")

    ev = evidence_rows(df_joined, preds, facility, why_year, None if why_zip=="All ZIPs" else why_zip)
    base_total = _sum_int_nonneg(ev.query("SOURCE=='actual'")["VALUE"])
    fc_total   = _sum_int_nonneg(ev.query("SOURCE=='forecast'")["VALUE"])
    st.write(f"**Totals for {facility} – {why_zip if why_zip!='All ZIPs' else 'All ZIPs'} – {why_year}**")
    st.write(f"- Actual: **{base_total:,}**  |  Forecast: **{fc_total:,}**")
    st.caption("Rows below show the individual contributions that roll up to the point above.")
    st.dataframe(ev, use_container_width=True)

# =============== CTA FOOTER ===============
st.markdown("---")
current_year = datetime.now().year
st.markdown(
    """
    <div style="padding:18px;border:1px solid #e5e7eb;border-radius:12px;background:#F9FAFB;">
      <h3 style="margin:0 0 8px 0;">Ready to pressure-test your OBBB exposure?</h3>
      <p style="margin:0 0 8px 0;">
        <strong>CALIBER360 Healthcare AI</strong> delivers facility-specific reviews that quantify
        OBBB risk by service line (AS, Inpatient, ED) by ZIP codes, stress-test payer-mix shifts, and
        scenario-model revenue with your data.
      </p>
      <p style="margin:0 0 12px 0;">
        <em>Advisory capacity is limited.</em> Secure a tailored review this month to align strategy,
        protect margins, and move from “what-ifs” to actionable plans.
      </p>
      <a href="https://caliber360ai.com" target="_blank"
         style="display:inline-block;padding:10px 14px;border-radius:8px;
                background:#2563eb;color:#fff;text-decoration:none;font-weight:600;">
         Book a tailored review →</a>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(f"© {current_year} CALIBER360 Healthcare AI. All rights reserved.")
