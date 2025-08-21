import os
import pandas as pd
import streamlit as st
from pathlib import Path

# --- Logo top-right (PNG) ---
st.set_page_config(page_title="Stealth Med RWEye — POC", layout="wide")

# ---- CLEAN CSS (remove all previous CSS blocks you added before) ----
st.markdown("""
<style>
/* Wider canvas + safe top padding so nothing clips */
.main .block-container {
    max-width: 1700px;
    padding-top: 3.2rem;      /* bump this up if you still see clipping */
    padding-bottom: 0.6rem;
}
/* Sidebar width */
[data-testid="stSidebar"] { width: 280px; }

/* Tight but safe header spacing */
h1 { margin: 0.25rem 0 0.9rem 0; }
</style>
""", unsafe_allow_html=True)

# ---- HEADER ROW: title left, logo right (NO fixed positioning) ----
logo_path = Path(__file__).parent / "logo.png"
header_left, header_right = st.columns([8, 2], gap="small")
with header_left:
    st.markdown("<h1>Stealth Med RWEye — POC</h1>", unsafe_allow_html=True)
with header_right:
    # set a fixed width so it doesn't overgrow; increase/decrease as you wish
    st.image(str(logo_path), width=220)

# ---------- paths ----------
BASE = "/Users/rahurkar.1/Library/CloudStorage/OneDrive-TheOhioStateUniversityWexnerMedicalCenter/FAERS/drug_id_platform"
ATC_PATH = os.path.join(BASE, "pedpubs_atc_merged.csv")
PRR_PATH = os.path.join(BASE, "prr_matched_with_cui_atc.csv")

#st.set_page_config(page_title="Stealth Med RWEye (POC)", layout="wide")

@st.cache_data
def load_data():
    atc = pd.read_csv(ATC_PATH, dtype=str).fillna("")
    # publication count
    if "unique_pub_count" in atc.columns:
        atc["unique_pub_count"] = pd.to_numeric(atc["unique_pub_count"], errors="coerce").fillna(0).astype(int)
    else:
        atc["unique_pub_count"] = 0

    prr = pd.read_csv(PRR_PATH, dtype=str).fillna("")
    if "PRR" in prr.columns:
        prr["PRR"] = pd.to_numeric(prr["PRR"], errors="coerce")
    return atc, prr

# Map our app's logical names to whatever your CSV has
def resolve_schema(df: pd.DataFrame):
    def pick(*cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    return {
        "L1_code": pick("L1_code", "level_1"),
        "L1_name": pick("L1_name", "level_1_desc"),
        "L2_code": pick("L2_code", "level_2"),
        "L2_name": pick("L2_name", "level_2_desc"),
        "L3_code": pick("L3_code", "level_3"),
        "L3_name": pick("L3_name", "level_3_desc"),
        "L4_code": pick("L4_code", "level_4"),
        "L4_name": pick("L4_name", "level_4_desc"),
        "drug_name": pick("rxnorm_name", "drug_name", "name", "preferred_name"),
    }

def codes(df, code_col, name_col, filters=None):
    d = df.copy()
    if filters:
        for col, val in filters.items():
            if val:
                d = d[d[col] == val]
    if not code_col or not name_col:
        return pd.DataFrame(columns=["code","name"])
    out = d[[code_col, name_col]].drop_duplicates()
    out.columns = ["code", "name"]
    return out.sort_values(["code","name"])

# ---------- load ----------
atc, prr = load_data()
schema = resolve_schema(atc)

# fail-fast helpful message if the file doesn’t have the expected fields
needed = ["L1_code","L1_name","L2_code","L2_name","L3_code","L3_name","L4_code","L4_name"]
if not all(schema[k] for k in needed):
    st.error(
        "Your CSV columns don’t match what the app expects. "
        "I looked for either L*_code/L*_name **or** level_* / level_*_desc.\n\n"
        f"Columns found: {list(atc.columns)}"
    )
    st.stop()

# drug display column
drug_col = schema["drug_name"] if schema["drug_name"] else "cui"
if drug_col not in atc.columns:
    atc[drug_col] = atc["cui"]

# ---------- sidebar slicers (pretty labels) ----------
st.sidebar.markdown("### Explore drugs by:")
st.sidebar.caption(
    "ATC hierarchy — L1 letter → L2 2-digit → L3 3-char → L4 4-char.\n"
    "Pick any level; lower levels will filter accordingly."
)

l1_df = codes(atc, schema["L1_code"], schema["L1_name"])
l1 = st.sidebar.selectbox(
    "L1 — Anatomical main group",
    [""] + l1_df["code"].tolist(),
    format_func=lambda c: f"{c} – {l1_df.loc[l1_df['code']==c,'name'].values[0]}" if c else "(all)",
    help="Top-level organ/system (e.g., N = Nervous system)"
)

l2_df = codes(atc, schema["L2_code"], schema["L2_name"], {schema["L1_code"]: l1} if l1 else None)
l2 = st.sidebar.selectbox(
    "L2 — Therapeutic main group",
    [""] + l2_df["code"].tolist(),
    format_func=lambda c: f"{c} – {l2_df.loc[l2_df['code']==c,'name'].values[0]}" if c else "(all)",
    help="Therapeutic main group within the L1 system"
)

l3_df = codes(atc, schema["L3_code"], schema["L3_name"],
              {schema["L1_code"]: l1, schema["L2_code"]: l2} if l2 else ({schema["L1_code"]: l1} if l1 else None))
l3 = st.sidebar.selectbox(
    "L3 — Pharmacological subgroup",
    [""] + l3_df["code"].tolist(),
    format_func=lambda c: f"{c} – {l3_df.loc[l3_df['code']==c,'name'].values[0]}" if c else "(all)",
    help="Pharmacological subgroup within L2"
)

l4_df = codes(atc, schema["L4_code"], schema["L4_name"], {
    schema["L1_code"]: l1 if l1 else None,
    schema["L2_code"]: l2 if l2 else None,
    schema["L3_code"]: l3 if l3 else None
})
l4 = st.sidebar.selectbox(
    "L4 — Chemical/Therapeutic/Pharmacological subgroup",
    [""] + l4_df["code"].tolist(),
    format_func=lambda c: f"{c} – {l4_df.loc[l4_df['code']==c,'name'].values[0]}" if c else "(all)",
    help="CTP subgroup; one step above chemical substance (L5)"
)

top_n = st.sidebar.number_input("Show top N drugs", min_value=10, max_value=500, value=100, step=10)

# ---------- filter atc table ----------
flt = atc.copy()
if l1: flt = flt[flt[schema["L1_code"]] == l1]
if l2: flt = flt[flt[schema["L2_code"]] == l2]
if l3: flt = flt[flt[schema["L3_code"]] == l3]
if l4: flt = flt[flt[schema["L4_code"]] == l4]

# ---------- drug list (one row per cui/name) ----------
drug_list = (
    flt.groupby(["cui", drug_col], as_index=False)
       .agg(unique_pub_count=("unique_pub_count", "max"))
       .sort_values("unique_pub_count", ascending=False)
       .head(top_n)
)

def link_for(cui, name):
    disp = name if isinstance(name, str) and name.strip() else cui
    return f"[{disp}](?cui={cui})"

drug_list_display = drug_list.assign(link=lambda d: d.apply(lambda r: link_for(r["cui"], r[drug_col]), axis=1))
drug_list_display = drug_list_display[["link", "cui", drug_col, "unique_pub_count"]]
drug_list_display = drug_list_display.rename(columns={drug_col: "drug_name"})

# ---------- layout ----------
#st.title("Stealth Med RWEye — POC")
left, right = st.columns((3.5,1))

with left:
    st.subheader("Drugs (click to drill down)")
    st.dataframe(
        drug_list_display,
        use_container_width=True,
        hide_index=True,
        height=800,  # taller table, fewer scrollbars
    )
    st.download_button("Download drug list (CSV)", drug_list.to_csv(index=False).encode("utf-8"),
                       file_name="drug_list.csv", mime="text/csv")

# selected cui from query param
sel_cui = st.query_params.get("cui", [None])
sel_cui = sel_cui[0] if isinstance(sel_cui, list) else sel_cui

with right:
    if sel_cui:
        st.subheader("Selected drug")
        hdr_cols = [c for c in [drug_col, schema["L1_code"], schema["L1_name"], schema["L2_code"], schema["L2_name"], schema["L3_code"], schema["L3_name"], schema["L4_code"], schema["L4_name"]] if c in atc.columns]
        hdr = atc[atc["cui"] == sel_cui][hdr_cols].drop_duplicates()
        st.write(hdr.iloc[0].to_dict() if not hdr.empty else {"cui": sel_cui})

# ---------- PRR detail ----------
if sel_cui:
    st.markdown("---")
    st.subheader("ADEs & PRRs for selected drug")
    view = prr[prr["cui"] == sel_cui].copy()
    keep = [c for c in ["drug","ADE","Age","Gender","A","B","C","D","PRR","match_type"] if c in view.columns]
    view = view[keep].sort_values("PRR", ascending=False, na_position="last")
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Download ADE/PRR (CSV)", view.to_csv(index=False).encode("utf-8"),
                       file_name=f"{sel_cui}_ade_prr.csv", mime="text/csv")
