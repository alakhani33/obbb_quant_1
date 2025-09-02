# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import gdown
import re  # if you use any regex helpers elsewhere

# Rural Transformation Funds slider settings
RTR_MIN   = -100_000_000
RTR_MAX   = 100_000_000
RTR_STEP  = 1_000_000
RTR_DEFAULT = 0  # default at zero

# Replace with your actual Google Drive file IDs
DF_JOINED_FILE_ID = "1vJfS10pDpAldKd1MhHptQZhrq5IcQxu_"   # e.g. '1AbC...'
PREDS_FILE_ID     = "1gbgrifQ0tMoKTIzAENnG9C9Vi-FyW7e8"

DATA_DIR = Path("data")  # local cache folder
DATA_DIR.mkdir(exist_ok=True)

def _ensure_drive_file(file_id: str, local_path: Path) -> Path:
    """Download from Google Drive if local_path is missing/empty."""
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    url = f"https://drive.google.com/uc?id={file_id}"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(local_path), quiet=False)
    return local_path


def _read_table_auto(path: Path) -> pd.DataFrame:
    """Read CSV / CSV.GZ / Parquet based on extension."""
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".gz" or p.name.endswith(".csv.gz"):
        return pd.read_csv(p, compression="gzip", low_memory=False)
    return pd.read_csv(p, low_memory=False)


# ----------------- Config: edit these paths -----------------
# DF_JOINED_PATH = Path(r"C:\Users\19163\OneDrive\Desktop\HCAI_INTEL\CENSUS_MKTSHARE\NEW_REV\REV_1\df_joined.csv")
# PREDS_PATH     = Path(r"C:\Users\19163\OneDrive\Desktop\HCAI_INTEL\CENSUS_MKTSHARE\NEW_REV\REV_1\preds.csv")

EXOG_COLS = [
    "median_household_income","population","median_age","per_capita_income",
    "poverty_count","poverty_universe","medicaid_ins_total","medicare_ins_total",
    "priv_ins_total","no_ins_total"
]
TARGET_COL = "DISCHARGES"

# Rural Transformation Relief slider settings
# RTR_MIN   = -1_000_000
# RTR_MAX   = 100_000_000
# RTR_STEP  = 1_000_000

def _default_mid(minv: int, maxv: int, step: int) -> int:
    """Midpoint of [minv, maxv], rounded to the nearest 'step' from minv."""
    mid = (minv + maxv) / 2.0
    return int(minv + round((mid - minv) / step) * step)

# RTR_DEFAULT = _default_mid(RTR_MIN, RTR_MAX, RTR_STEP)  # -> $50,000,000

# ----------------- Streamlit setup -----------------
# st.set_page_config(
#     page_title="CALIBER360 OBBB Quantification Layer",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )
# st.title("CALIBER360 OBBB Quantification Layer")
# st.caption(
#     "Discharges include **Ambulatory Surgery (AS) Only**, **Inpatient Only**, "
#     "**Emergency Department (ED) Only**, and **Inpatient from ED**."
# )
st.title("CALIBER360 OBBB Quantification")

# Bold tagline (separate from caption so it pops)
st.markdown("**OBBB impact, quantified for every California hospital—by ZIP, Service Line, and Payer Mix.**")

st.caption(
    "Discharges include **Ambulatory Surgery (AS)**, **Inpatient**, "
    "**Emergency Department (ED)**, and **Inpatient from ED**."
)


# Scrollbars for long dropdown menus (facility + ZIP)
st.markdown(
    """
    <style>
    div[data-baseweb="menu"] {
        max-height: 320px !important;
        overflow-y: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- Helpers -----------------
def _normalize_zip(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"\.0$", "", regex=True)
         .str.zfill(5)
    )

def _int_nonneg(series: pd.Series) -> pd.Series:
    """Force whole, non-negative numbers."""
    arr = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy(dtype=float)
    arr = np.maximum(arr, 0.0)
    return pd.Series(np.rint(arr).astype(int), index=series.index)

def _fmt_pct_signed(v: int) -> str:
    return f"{v:+d}%"

def _fmt_money_signed(v: int | float) -> str:
    sign = "+" if v > 0 else ("-" if v < 0 else "")
    return f"{sign}${abs(v):,}"

# @st.cache_data(show_spinner=True)
# def load_df_joined_cached(path_str: str) -> pd.DataFrame:
#     p = Path(path_str)
#     if not p.exists():
#         raise FileNotFoundError(f"df_joined not found at: {p}")
#     df = pd.read_csv(p, low_memory=False, dtype={"PZIP": "string"})
#     df.columns = df.columns.str.strip()

#     req = {"FACILITY_NAME","PZIP","YEAR", TARGET_COL}
#     missing = req - set(df.columns)
#     if missing:
#         raise ValueError(f"`df_joined.csv` missing columns: {sorted(missing)}")

#     df["PZIP"] = _normalize_zip(df["PZIP"])
#     df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")

#     # Coerce numerics
#     for c in [TARGET_COL] + [c for c in EXOG_COLS if c in df.columns]:
#         df[c] = pd.to_numeric(df[c], errors="coerce")

#     # Handle sentinel NAs in exogs (if present)
#     NA_SENTINELS = [-666666666, -66666666, -999999999, -99999999]
#     present_exogs = [c for c in EXOG_COLS if c in df.columns]
#     if present_exogs:
#         df[present_exogs] = df[present_exogs].replace(NA_SENTINELS, np.nan)

#     df = df.dropna(subset=["YEAR"]).copy()
#     df["YEAR"] = df["YEAR"].astype(int)

#     # De-dup by FACILITY/PZIP/YEAR
#     df = (
#         df.sort_values(["FACILITY_NAME","PZIP","YEAR"])
#           .drop_duplicates(["FACILITY_NAME","PZIP","YEAR"], keep="last")
#           .reset_index(drop=True)
#     )
#     return df

# @st.cache_data(show_spinner=True)
# def load_preds_cached(path_str: str) -> pd.DataFrame:
#     p = Path(path_str)
#     if not p.exists():
#         raise FileNotFoundError(f"preds not found at: {p}")
#     df = pd.read_csv(p, low_memory=False, dtype={"PZIP": "string"})
#     df.columns = df.columns.str.strip()

#     req = {"FACILITY_NAME","PZIP","YEAR","PRED_DISCHARGES"}
#     missing = req - set(df.columns)
#     if missing:
#         raise ValueError(f"`preds.csv` missing columns: {sorted(missing)}")

#     df["PZIP"] = _normalize_zip(df["PZIP"])
#     df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
#     df["PRED_DISCHARGES"] = pd.to_numeric(df["PRED_DISCHARGES"], errors="coerce")

#     df = df.dropna(subset=["YEAR"]).copy()
#     df["YEAR"] = df["YEAR"].astype(int)

#     df = (
#         df.sort_values(["FACILITY_NAME","PZIP","YEAR"])
#           .drop_duplicates(["FACILITY_NAME","PZIP","YEAR"], keep="last")
#           .reset_index(drop=True)
#     )
#     return df
@st.cache_data(show_spinner=False)
def load_df_joined_from_drive(file_id: str) -> pd.DataFrame:
    # Download (if needed) into local cache
    local_path = DATA_DIR / "df_joined.csv.gz"   # keep .gz locally
    _ensure_drive_file(file_id, local_path)

    # Read gzip CSV
    df = pd.read_csv(local_path, compression="gzip", low_memory=False, dtype={"PZIP": "string"})
    df.columns = df.columns.str.strip()

    # Schema checks
    req = {"FACILITY_NAME", "PZIP", "YEAR", TARGET_COL}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"`df_joined.csv.gz` missing columns: {sorted(missing)}")

    # Normalize / validate ZIPs
    df["PZIP"] = (
        df["PZIP"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)
    )
    # Drop invalid 5-digit ZIPs and '00000'
    df = df[df["PZIP"].str.fullmatch(r"\d{5}") & (df["PZIP"] != "00000")].copy()

    # Year + numerics
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    for c in [TARGET_COL] + [c for c in EXOG_COLS if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sentinels → NaN for exogs
    NA_SENTINELS = [-666666666, -66666666, -999999999, -99999999]
    present_exogs = [c for c in EXOG_COLS if c in df.columns]
    if present_exogs:
        df[present_exogs] = df[present_exogs].replace(NA_SENTINELS, np.nan)

    # Clean year, dedupe
    df = df.dropna(subset=["YEAR"]).copy()
    df["YEAR"] = df["YEAR"].astype(int)
    df = (
        df.sort_values(["FACILITY_NAME", "PZIP", "YEAR"])
          .drop_duplicates(["FACILITY_NAME", "PZIP", "YEAR"], keep="last")
          .reset_index(drop=True)
    )
    return df


@st.cache_data(show_spinner=False)
def load_preds_from_drive(file_id: str) -> pd.DataFrame:
    # Download (if needed) into local cache
    local_path = DATA_DIR / "preds.csv.gz"       # keep .gz locally
    _ensure_drive_file(file_id, local_path)

    # Read gzip CSV
    df = pd.read_csv(local_path, compression="gzip", low_memory=False, dtype={"PZIP": "string"})
    df.columns = df.columns.str.strip()

    # Schema checks
    req = {"FACILITY_NAME", "PZIP", "YEAR", "PRED_DISCHARGES"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"`preds.csv.gz` missing columns: {sorted(missing)}")

    # Normalize / validate ZIPs
    df["PZIP"] = (
        df["PZIP"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)
    )
    df = df[df["PZIP"].str.fullmatch(r"\d{5}") & (df["PZIP"] != "00000")].copy()

    # Year + numerics
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    df["PRED_DISCHARGES"] = pd.to_numeric(df["PRED_DISCHARGES"], errors="coerce")

    # Clean year, dedupe
    df = df.dropna(subset=["YEAR"]).copy()
    df["YEAR"] = df["YEAR"].astype(int)
    df = (
        df.sort_values(["FACILITY_NAME", "PZIP", "YEAR"])
          .drop_duplicates(["FACILITY_NAME", "PZIP", "YEAR"], keep="last")
          .reset_index(drop=True)
    )
    return df


def build_series(df_joined: pd.DataFrame, preds: pd.DataFrame,
                 facility: str, selected_pzips: list[str]):
    """
    Returns (hist_2020_2024, fc_2025_2026, combined_series) — integers, non-negative.
    Aggregates DISCHARGES across selected_pzips for the chosen facility.
    Always treats 2024 as 'actual' anchor (solid line).
    """
    # Actuals up to 2024 (if available in df_joined)
    hist = (
        df_joined.query("FACILITY_NAME == @facility and 2020 <= YEAR <= 2024 and PZIP in @selected_pzips")
                 .groupby("YEAR", as_index=False)[TARGET_COL].sum()
                 .rename(columns={TARGET_COL: "VALUE"})
                 .sort_values("YEAR")
    )
    hist["VALUE"] = _int_nonneg(hist["VALUE"])

    # Forecasts 2024–2026 (use 2024 as anchor if not in hist)
    fc = (
        preds.query("FACILITY_NAME == @facility and 2024 <= YEAR <= 2026 and PZIP in @selected_pzips")
             .groupby("YEAR", as_index=False)["PRED_DISCHARGES"].sum()
             .rename(columns={"PRED_DISCHARGES": "VALUE"})
             .sort_values("YEAR")
    )
    fc["VALUE"] = _int_nonneg(fc["VALUE"])

    # Ensure 2024 is on the solid (Actual) line:
    if 2024 not in hist["YEAR"].tolist():
        fc_2024 = fc.loc[fc["YEAR"] == 2024]
        if not fc_2024.empty:
            hist_2020_2024 = pd.concat([hist, fc_2024], ignore_index=True).sort_values("YEAR")
        else:
            hist_2020_2024 = hist.copy()
    else:
        hist_2020_2024 = hist.copy()

    # Forecast segment begins at 2025
    fc_2025_2026 = fc.loc[fc["YEAR"] >= 2025].copy()
    fc_2025_2026["VALUE"] = _int_nonneg(fc_2025_2026["VALUE"])

    combined = pd.concat(
        [
            hist_2020_2024.assign(KIND="Actual (2020–2024)"),
            fc_2025_2026.assign(KIND="Forecast (2025–2026)")
        ],
        ignore_index=True
    ).sort_values("YEAR")

    return hist_2020_2024, fc_2025_2026, combined

def plot_series(facility: str, hist_2020_2024: pd.DataFrame, fc_2025_2026: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9,5))

    # Solid through 2024
    if not hist_2020_2024.empty:
        ax.plot(
            hist_2020_2024["YEAR"], hist_2020_2024["VALUE"],
            marker="o", linestyle="-", color="tab:blue", label="Actual (2020–2024)"
        )

    # Dashed 2025–2026, connected from the last actual point (2024)
    if not fc_2025_2026.empty:
        if not hist_2020_2024.empty:
            anchor = hist_2020_2024.tail(1)[["YEAR","VALUE"]]
            fc_line = pd.concat([anchor, fc_2025_2026], ignore_index=True)
        else:
            fc_line = fc_2025_2026
        ax.plot(
            fc_line["YEAR"], fc_line["VALUE"],
            marker="o", linestyle="--", color="tab:orange", label="Forecast (2025–2026)"
        )

    ax.set_xticks(list(range(2020, 2027)))
    ax.set_xlabel("Year")
    ax.set_ylabel("Discharges (total)")
    ax.set_title(f"{facility}: Discharges 2020–2026")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ----------------- Load data (cached) -----------------
# try:
#     # df_joined = load_df_joined_cached(str(DF_JOINED_PATH))
#     # preds     = load_preds_cached(str(PREDS_PATH))
#     df_joined = load_df_joined_from_drive(DF_JOINED_FILE_ID)
#     preds     = load_preds_from_drive(PREDS_FILE_ID)
# except Exception as e:
#     st.error(f"Failed to load data: {e}")
#     st.stop()

with st.spinner("CALIBER360 loading data for all CA hospitals — please hold on..."):
    try:
        df_joined = load_df_joined_from_drive(DF_JOINED_FILE_ID)
        preds     = load_preds_from_drive(PREDS_FILE_ID)
    except Exception as e:
        st.error(f"Data load failed: {e}")
        st.stop()

# ----------------- Sidebar — OBBB Impact (always visible) -----------------
with st.sidebar:
    st.subheader("OBBB Impact")
    medicaid_delta  = st.slider("Medicaid impact (%)",  -20, 20, 0, step=1)
    uninsured_delta = st.slider("Uninsured impact (%)", -20, 20, 0, step=1)
    medicare_delta  = st.slider("Medicare impact (%)",  -20, 20, 0, step=1)

    # Rural Transformation Relief — default centered on the dial
    # rtr_amount = st.slider(
    #     "Rural Transformation Funds (USD)",
    #     min_value=RTR_MIN,
    #     max_value=RTR_MAX,
    #     value=RTR_DEFAULT,   # centered default (≈ $50M)
    #     step=RTR_STEP,
    # )
    rtr_amount = st.slider(
    "Rural Transformation Funds (USD)",
    min_value=RTR_MIN,
    max_value=RTR_MAX,
    value=RTR_DEFAULT,   # sits at 0 by default
    step=RTR_STEP,
)


    # Generalized, compact summary for ALL dials
    # st.caption(
    #     "Current OBBB impact settings:  "
    #     f"Medicaid {_fmt_pct_signed(medicaid_delta)} · "
    #     f"Uninsured {_fmt_pct_signed(uninsured_delta)} · "
    #     f"Medicare {_fmt_pct_signed(medicare_delta)} · "
    #     f"Rural Relief {_fmt_money_signed(rtr_amount)}.  "
    #     "_Not applied to this chart; will be used in the revenue model._"
    # )
    st.caption(
    "Current OBBB impact settings:  "
    f"Medicaid {_fmt_pct_signed(medicaid_delta)} · "
    f"Uninsured {_fmt_pct_signed(uninsured_delta)} · "
    f"Medicare {_fmt_pct_signed(medicare_delta)} · "
    f"Rural Funds {_fmt_money_signed(rtr_amount)}.  "
    "_Not applied to this chart; available in revenue model._"
)

    # Persist for downstream modules
    st.session_state.update({
        "knob_medicaid_pct": medicaid_delta,
        "knob_uninsured_pct": uninsured_delta,
        "knob_medicare_pct": medicare_delta,
        "knob_rural_relief_usd": rtr_amount,
    })

# ----------------- Main UI controls -----------------
# Facility selector
facilities_all = sorted(set(df_joined["FACILITY_NAME"].unique()).union(set(preds["FACILITY_NAME"].unique())))
facility = st.selectbox("Facility", facilities_all, index=0, help="Type to search, or scroll the list.")

# ZIP selector (+ 'All ZIPs')
pzips_for_fac = sorted(set(
    df_joined.loc[df_joined["FACILITY_NAME"] == facility, "PZIP"].unique()
).union(
    preds.loc[preds["FACILITY_NAME"] == facility, "PZIP"].unique()
))

# remove invalids / placeholders just in case
pzips_for_fac = [
    z for z in pzips_for_fac
    if isinstance(z, str) and len(z) == 5 and z.isdigit() and z != "00000"
]

zip_options = ["All ZIPs"] + pzips_for_fac

zip_options = ["All ZIPs"] + pzips_for_fac
zip_choice = st.selectbox("ZIP code (PZIP)", options=zip_options, index=0, help="Type to search, or scroll the list.")
selected_pzips = pzips_for_fac if zip_choice == "All ZIPs" else [zip_choice]

# ----------------- Plot -----------------
if not pzips_for_fac:
    st.warning("No ZIPs found for this facility in the loaded data.")
else:
    hist_2020_2024, fc_2025_2026, combined = build_series(
        df_joined, preds, facility, selected_pzips
    )
    if hist_2020_2024.empty and fc_2025_2026.empty:
        st.warning("No data available for the selected filters.")
    else:
        plot_series(facility, hist_2020_2024, fc_2025_2026)

st.markdown("---")
st.subheader("Data used for the chart")
if "hist_2020_2024" in locals() and "fc_2025_2026" in locals():
    data_to_show = pd.concat(
        [
            hist_2020_2024.assign(_segment="Actual (2020–2024)"),
            fc_2025_2026.assign(_segment="Forecast (2025–2026)")
        ],
        ignore_index=True
    ).rename(columns={"VALUE": "Discharges"}).sort_values(["_segment","YEAR"])
    data_to_show["Discharges"] = _int_nonneg(data_to_show["Discharges"])
    st.dataframe(data_to_show, use_container_width=True)

# ----------------- CTA Footer + Copyright -----------------
st.markdown("---")
current_year = datetime.now().year
st.markdown(
    """
    <div style="padding:18px;border:1px solid #e5e7eb;border-radius:12px;background:#F9FAFB;">
      <h3 style="margin:0 0 8px 0;">Ready to pressure-test your OBBB exposure?</h3>
      <p style="margin:0 0 8px 0;">
        <strong>CALIBER360 Healthcare AI</strong> delivers facility-specific reviews that quantify
        OBBB risk by service line (AS, Inpatient, ED) by zip codes, stress-test payer-mix shifts, and
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
