# app.py
# Streamlit POC for Stealth Med RWEye — shows drugs by ATC with pubs + Rx volume
# Drop-in replacement. No notebook work required.

import pandas as pd
import streamlit as st
from pathlib import Path

# -------------------------
# Paths (edit BASE if needed)
# -------------------------
BASE = Path("/Users/rahurkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversityWexnerMedicalCenter/FAERS/drug_id_platform")

PUBS_FN = BASE / "pedpubs_atc_merged.csv"        # pubs + ATC + CUI
RX_FN   = BASE / "rx_vol_joined_to_umls.csv"     # Rx volume joined to UMLS (from your notebook)
LOGO_FN = BASE / "logo2.png"                      # optional logo

#st.set_page_config(page_title="Stealth Med RWEye — POC", layout="wide")

# --- Full-width banner logo (single CSS block)
st.markdown("""
<style>
/* widen content area and add safe top padding so nothing clips under the app header */
.main .block-container { 
    max-width: 1700px;
    padding-top: 3.25rem;   /* adjust 2.5–4rem to taste */
    padding-bottom: 0.8rem;
}
/* do NOT zero-out the first child's margin; that can cause perceived clipping */
</style>
""", unsafe_allow_html=True)

if LOGO_FN.exists():
    st.image(str(LOGO_FN), use_column_width=True)



# -------------------------
# Data loading / shaping
# -------------------------
@st.cache_data
def load_data(pubs_fn: Path, rx_fn: Path) -> pd.DataFrame:
    # 1) Publications table
    pubs = pd.read_csv(pubs_fn)

    # Ensure we have real drug names
    if "rxnorm_name" in pubs.columns:
        pubs = pubs[pubs["rxnorm_name"].notna() & (pubs["rxnorm_name"].str.strip() != "")]
    else:
        pubs["rxnorm_name"] = ""   # safety

    # Keep stable schema
    keep_cols = [
        "cui","rxnorm_tty","rxnorm_name",
        "L1_code","L1_name","L2_code","L2_name","L3_code","L3_name","L4_code","L4_name",
        "unique_pub_count"
    ]
    keep_cols = [c for c in keep_cols if c in pubs.columns]
    pubs = pubs[keep_cols].copy()

    # Aggregate pubs at CUI + L1..L4 (in case of multiple L5 rows)
    group_keys = [k for k in ["cui","rxnorm_name","rxnorm_tty",
                              "L1_code","L1_name","L2_code","L2_name",
                              "L3_code","L3_name","L4_code","L4_name"] if k in pubs.columns]
    pubs_agg = (pubs.groupby(group_keys, as_index=False)["unique_pub_count"].sum()
                   if "unique_pub_count" in pubs.columns else
                   pubs.assign(unique_pub_count=0))

    # 2) Rx volume table
    rx = pd.read_csv(rx_fn)

    # Guess the total-volume column name
    vol_col = None
    for candidate in ["rx_freq_total", "freq_total", "freq"]:
        if candidate in rx.columns:
            vol_col = candidate
            break
    if vol_col is None:
        rx["rx_freq_total"] = 0
        vol_col = "rx_freq_total"

    # Keys to align on
    rx_keys = [k for k in ["cui","L1_code","L1_name","L2_code","L2_name","L3_code","L3_name","L4_code","L4_name"] if k in rx.columns]
    rx = rx[rx_keys + [vol_col]].copy()

    # Merge pubs + rx
    df = pubs_agg.merge(rx, on=rx_keys, how="left")
    df[vol_col] = df[vol_col].fillna(0).astype(int)

    # Friendly names
    df = df.rename(columns={
        "unique_pub_count": "pub_count",
        vol_col: "rx_volume",
        "rxnorm_name": "drug_name"
    })

    # For consistent filtering, fill missing ATC codes/names with "(unknown)"
    for c in ["L1_code","L1_name","L2_code","L2_name","L3_code","L3_name","L4_code","L4_name"]:
        if c in df.columns:
            df[c] = df[c].fillna("(unknown)")

    return df

df = load_data(PUBS_FN, RX_FN)

# -------------------------
# Header + Logo
# -------------------------
#col_title, col_spacer = st.columns([0.9, 0.1])
#with col_title:
#    st.markdown("## **Stealth Med RWEye — POC**")
#
#with col_spacer:
#    if LOGO_FN.exists():
#        st.image(str(LOGO_FN), width=800)

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.markdown("### Explore drugs by:")
st.sidebar.caption("ATC hierarchy → L1 letter → L2 2-digit → L3 3-char → L4 4-char. Select a level; lower levels filter accordingly.")

def optlist(code_col, name_col):
    if code_col not in df.columns or name_col not in df.columns:
        return ["(all)"], "(all)"
    opts = ["(all)"] + sorted(
        {f"{c} — {n}" for c, n in zip(df[code_col], df[name_col]) if pd.notna(c) and pd.notna(n)}
    )
    return opts, "(all)"

l1_opts, l1_default = optlist("L1_code", "L1_name")
l2_opts, l2_default = optlist("L2_code", "L2_name")
l3_opts, l3_default = optlist("L3_code", "L3_name")
l4_opts, l4_default = optlist("L4_code", "L4_name")

l1_sel = st.sidebar.selectbox("L1 — Anatomical main group", l1_opts, index=l1_opts.index(l1_default))
l2_sel = st.sidebar.selectbox("L2 — Therapeutic main group", l2_opts, index=l2_opts.index(l2_default))
l3_sel = st.sidebar.selectbox("L3 — Pharmacological subgroup", l3_opts, index=l3_opts.index(l3_default))
l4_sel = st.sidebar.selectbox("L4 — Chemical/Therapeutic/Pharmacological subgroup", l4_opts, index=l4_opts.index(l4_default))

metric = st.sidebar.radio("Rank by:", ["Publications", "Rx volume"], horizontal=True, index=0)
metric_col = "pub_count" if metric == "Publications" else "rx_volume"

top_n = st.sidebar.number_input("Show top N drugs", min_value=10, max_value=1000, value=100, step=10)

# -------------------------
# Filtering
# -------------------------
filtered = df.copy()

def code_from_label(label: str) -> str | None:
    if not label or label == "(all)":
        return None
    # labels look like "N06 — Psychoanaleptics"
    return label.split(" — ", 1)[0].strip()

for level, sel in [("L1_code", l1_sel), ("L2_code", l2_sel), ("L3_code", l3_sel), ("L4_code", l4_sel)]:
    code = code_from_label(sel)
    if code and level in filtered.columns:
        filtered = filtered[filtered[level] == code]

# -------------------------
# Aggregate + display
# -------------------------
group_cols = [c for c in ["cui","drug_name","L1_code","L1_name","L2_code","L2_name","L3_code","L3_name","L4_code","L4_name"] if c in filtered.columns]
agg = (filtered
       .groupby(group_cols, as_index=False)
       .agg(pub_count=("pub_count","sum"), rx_volume=("rx_volume","sum"))
       .sort_values(metric_col, ascending=False)
       .head(int(top_n)))

st.markdown("### Drugs (click to drill down)")
st.dataframe(agg, use_container_width=True, hide_index=True)

# -------------------------
# Download
# -------------------------
csv_bytes = agg.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download drug list (CSV)",
    data=csv_bytes,
    file_name="drug_list.csv",
    mime="text/csv",
)
