# app.py — smr3/prr3 schema, is_first filter, dropdown ADE, CSV downloads
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Stealth Med RWEye", page_icon="💊", layout="wide")

# -----------------------
# Branding (logo only in main content, left-aligned)
# -----------------------
# Defaults to your new path but still works if logo.svg sits next to app.py
DEFAULT_DIR = Path("/Users/rahurkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversityWexnerMedicalCenter/FAERS/drug_id_platform/poc")
LOGO_PATH = Path(os.getenv("LOGO_SVG", DEFAULT_DIR / "logo.svg"))
if not LOGO_PATH.exists():
    # fallback to local folder if env/default missing
    LOGO_PATH = Path(__file__).parent / "logo.svg"

def render_header_logo():
    if LOGO_PATH.exists():
        left, right = st.columns([1, 7])   # tweak the ratio if you want
        with left:
            try:
                st.image(str(LOGO_PATH), width=600)  # size tweak here (e.g., 120–160)
            except TypeError:
                st.image(str(LOGO_PATH), width=600)

# -----------------------
# Paths (override with env if you want)
# -----------------------
SMR_PATH = Path(os.getenv("SMR3_CSV", DEFAULT_DIR / "smr3.csv"))
PRR_PATH = Path(os.getenv("PRR3_CSV", DEFAULT_DIR / "prr3.csv"))

# -----------------------
# Loaders
# -----------------------
@st.cache_data(show_spinner=True)
def load_smr(pathlike) -> pd.DataFrame:
    df = pd.read_csv(pathlike, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    # normalize strings
    for c in ["drug", "agegroup", "l1", "l2", "l3", "l4", "cui"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # numeric
    for c in ["prescriptions", "pubs", "is_first"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # enforce your rule: only keep rows where is_first == 1
    if "is_first" in df.columns:
        df = df[df["is_first"] == 1]

    # treat empty strings as NA for ATC
    for c in ["l1", "l2", "l3", "l4"]:
        if c in df.columns:
            df[c] = df[c].replace({"": np.nan, "nan": np.nan})

    return df

@st.cache_data(show_spinner=True)
def load_prr(pathlike) -> pd.DataFrame:
    df = pd.read_csv(pathlike, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    # normalize strings
    for c in ["drug", "cui", "agegroup", "pt"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    # numeric
    for c in ["prr", "ror", "ic", "ebgm"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_or_upload(label: str, default_path: Path, fn):
    try:
        return fn(default_path), str(default_path)
    except Exception as e:
        st.warning(f"Could not open **{label}** at `{default_path}`. Upload below. Error: {e}")
        up = st.file_uploader(f"Upload {label}", type=["csv"], key=f"up_{label}")
        if up is not None:
            return fn(up), f"(uploaded) {label}"
        return pd.DataFrame(), f"(missing) {label}"

smr, smr_src = load_or_upload("smr3.csv", SMR_PATH, load_smr)
prr, prr_src = load_or_upload("prr3.csv", PRR_PATH, load_prr)
if smr.empty or prr.empty:
    st.stop()

# -----------------------
# Sidebar (filters)
# -----------------------
with st.sidebar:
    st.title("Filters")

# Age radio — your smr3 has: 0-2, 3-10, 11-17, Total
age_order = ["0-2", "3-10", "11-17", "Total"]
age_available = [a for a in age_order if a in smr["agegroup"].dropna().unique().tolist()] or age_order
age_choice = st.sidebar.radio("Age category", age_available, index=min(len(age_available)-1, 3))

def opts(series: pd.Series):
    vals = [x for x in series.dropna().unique().tolist() if str(x).strip() != ""]
    return ["All"] + sorted(vals)

# Cascading ATC (use the full string from smr3, e.g., "J05, ANTIVIRALS FOR SYSTEMIC USE")
l1_choice = st.sidebar.selectbox("Anatomical main group (ATC L1)", opts(smr["l1"]) if "l1" in smr.columns else ["All"])
smr_l1 = smr if l1_choice == "All" else smr[smr["l1"] == l1_choice]

l2_choice = st.sidebar.selectbox("Therapeutic main group (ATC L2)", opts(smr_l1["l2"]) if "l2" in smr_l1.columns else ["All"])
smr_l2 = smr_l1 if l2_choice == "All" else smr_l1[smr_l1["l2"] == l2_choice]

l3_choice = st.sidebar.selectbox("Pharmacological subgroup (ATC L3)", opts(smr_l2["l3"]) if "l3" in smr_l2.columns else ["All"])
smr_l3 = smr_l2 if l3_choice == "All" else smr_l2[smr_l2["l3"] == l3_choice]

l4_choice = st.sidebar.selectbox("Chemical subgroup (ATC L4)", opts(smr_l3["l4"]) if "l4" in smr_l3.columns else ["All"])
smr_filtered = smr_l3 if l4_choice == "All" else smr_l3[smr_l3["l4"] == l4_choice]

# Drug search (autocomplete)
drug_options = sorted(smr_filtered["drug"].dropna().unique().tolist())
search_pick = st.sidebar.multiselect("Search drug", options=drug_options)

# Optional ATC display columns (your “Only show ATC rows if selected”)
optional_atc_cols = [c for c in ["l1", "l2", "l3", "l4"] if c in smr_filtered.columns]
selected_optional = st.sidebar.multiselect("Add ATC columns", options=optional_atc_cols)

# Sorting
sort_metric = st.sidebar.radio("Sort by", ["Prescriptions", "Publications"], index=0)
sort_desc = st.sidebar.checkbox("Sort descending", value=True)

# -----------------------
# Apply filters
# -----------------------
df = smr_filtered.copy()
df = df[df["agegroup"] == age_choice]
if search_pick:
    df = df[df["drug"].isin(search_pick)]

# Base display fields
df["Prescriptions"] = df["prescriptions"]

def pubs_display(val):
    if pd.isna(val) or float(val) == 0.0:
        return "No Pubs in MPRINT"
    try:
        v = float(val)
        return f"{int(v)}" if v.is_integer() else f"{v}"
    except Exception:
        return str(val)

df["Publications_num"] = df["pubs"]
df["Publications"] = df["pubs"].apply(pubs_display)

# Sort
if sort_metric == "Publications":
    df = df.sort_values(by="Publications_num", ascending=not sort_desc, na_position="last")
else:
    df = df.sort_values(by="Prescriptions", ascending=not sort_desc, na_position="last")

# -----------------------
# Results (logo + table + CSV download)
# -----------------------
render_header_logo()
st.markdown("### Results")

base_cols = ["drug", "agegroup", "Prescriptions", "Publications"]
show_cols = base_cols + [c for c in selected_optional if c in df.columns]  # ATC only if selected
results_df = df[show_cols]

st.dataframe(results_df, use_container_width=True, hide_index=True)

# Download results CSV
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
res_fname = f"rweeye_results_{age_choice.replace(' ', '_')}_{ts}.csv"
st.download_button(
    label="⬇️ Download results CSV",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name=res_fname,
    mime="text/csv",
)

# -----------------------
# ADE drill-down (dropdown selection) + CSV download
# -----------------------
visible_drugs = results_df["drug"].dropna().unique().tolist()
pick = st.selectbox("Select a drug for ADE metrics", ["(None)"] + visible_drugs)
selected_drug = None if pick == "(None)" else pick

st.markdown("---")
st.markdown("### ADE metrics")

if selected_drug:
    # Try to find CUI for the selected drug from smr3 (any age row is fine)
    cui_val = (
        smr.loc[smr["drug"] == selected_drug, "cui"]
        .dropna()
        .astype(str)
        .str.strip()
        .head(1)
        .tolist()
    )
    cui_val = cui_val[0] if cui_val else None

    if age_choice == "Total":
        # Show all pediatric ADE rows for that drug
        if cui_val:
            ade_df = prr[prr["cui"] == cui_val].copy()
        else:
            ade_df = prr[prr["drug"].str.casefold() == selected_drug.casefold()].copy()
        subtitle = f"{selected_drug} — All pediatric ages"
    else:
        if cui_val:
            ade_df = prr[(prr["cui"] == cui_val) & (prr["agegroup"] == age_choice)].copy()
        else:
            ade_df = prr[(prr["drug"].str.casefold() == selected_drug.casefold()) & (prr["agegroup"] == age_choice)].copy()
        subtitle = f"{selected_drug} — {age_choice}"

    if ade_df.empty:
        st.info(f"No ADE data for **{subtitle}**.")
    else:
        # prr3 columns: pt, prr, ror, ic, ebgm (and agegroup)
        cols_to_show = [c for c in ["agegroup", "pt", "prr", "ror", "ic", "ebgm"] if c in ade_df.columns]
        if "prr" in ade_df.columns:
            ade_df = ade_df.sort_values("prr", ascending=False, na_position="last")
        st.markdown(f"#### {subtitle}")
        ade_view = ade_df[cols_to_show]
        st.dataframe(ade_view, use_container_width=True, hide_index=True)

        # Download ADE CSV
        ade_fname = f"rweeye_ade_{selected_drug.replace(' ', '_')}_{age_choice.replace(' ', '_')}_{ts}.csv"
        st.download_button(
            label="⬇️ Download ADE CSV",
            data=ade_view.to_csv(index=False).encode("utf-8"),
            file_name=ade_fname,
            mime="text/csv",
        )
else:
    st.caption("Pick a drug above to view ADE metrics (PRR / ROR / IC / EBGM).")

# -----------------------
# Debug info
# -----------------------
with st.expander("ⓘ Data info / Debug"):
    st.write(f"SMR3 path: {smr_src}")
    st.write(f"PRR3 path: {prr_src}")
    st.write("SMR rows (after is_first==1):", len(smr))
    st.write("PRR rows:", len(prr))
    st.write("Filtered rows:", len(df))
