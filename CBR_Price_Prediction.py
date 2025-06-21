import streamlit as st
import pandas as pd
import numpy as np
import random

# === Fungsi Generate Data ===
def generate_cases(property_type, num_cases, start_id):
    cases = []
    regions = ["Kabupaten Sleman", "Kota Yogyakarta", "Kabupaten Bantul", "Kabupaten Kulon Progo", "Kabupaten Gunung Kidul"]
    region_multiplier = {
        "Kabupaten Sleman": 1.00,
        "Kota Yogyakarta": 1.20,
        "Kabupaten Bantul": 0.90,
        "Kabupaten Kulon Progo": 0.80,
        "Kabupaten Gunung Kidul": 0.85,
    }

    for i in range(num_cases):
        case_id = start_id + i
        region = random.choice(regions)
        mult = region_multiplier[region]
        case = {"id": case_id, "property_type": property_type, "region": region}
        detail = {}
        base_total = 0

        if property_type == "apartment":
            sqm = random.randint(20, 80)
            base_total = 4_000_000 * sqm
            detail = {
                "sqm": sqm, "lot_size": 0,
                "bedrooms": random.randint(1, 3),
                "bathrooms": random.randint(1, 2),
                "year_built": random.randint(1990, 2020),
                "floor_number": random.randint(1, 25),
                "amenity_score": random.randint(0, 5),
            }
            base_total += detail["floor_number"] * 50_000 + detail["amenity_score"] * 200_000

        elif property_type == "non-clustered house":
            sqm = random.randint(80, 300)
            lot = random.randint(100, 600)
            base_total = 3_000_000 * sqm + 500_000 * lot
            detail = {
                "sqm": sqm, "lot_size": lot,
                "bedrooms": random.randint(2, 5),
                "bathrooms": random.randint(1, 4),
                "year_built": random.randint(1980, 2018),
            }

        elif property_type == "land":
            sqm_land = random.randint(200, 2000)
            base_total = 1_000_000 * sqm_land
            detail = {
                "sqm": 0, "lot_size": sqm_land,
                "zoning": random.choice(["residential", "commercial", "agricultural"]),
                "access_to_road": random.choice([True, False]),
            }
            if detail["zoning"] == "commercial": base_total *= 1.5
            if detail["access_to_road"]: base_total *= 1.2

        elif property_type == "clustered house":
            sqm = random.randint(60, 200)
            lot = random.randint(50, 300)
            base_total = 3_500_000 * sqm + 300_000 * lot
            detail = {
                "sqm": sqm, "lot_size": lot,
                "bedrooms": random.randint(2, 4),
                "bathrooms": random.randint(2, 3),
                "year_built": random.randint(2000, 2022),
                "cluster_amenity_score": random.randint(0, 5),
            }
            base_total += detail["cluster_amenity_score"] * 300_000

        noisy_total = base_total * random.uniform(0.9, 1.1) * mult
        denominator_sqm = detail.get("sqm", 0)
        if not isinstance(denominator_sqm, (int, float)) or denominator_sqm <= 0:
            denominator_sqm = detail.get("lot_size", 1)
        if denominator_sqm == 0: denominator_sqm = 1

        unit_price = noisy_total / denominator_sqm
        case["sale_price_per_sqm"] = int(unit_price)
        case.update(detail)
        cases.append(case)
    return cases

# === Gower & CBR ===
def calculate_feature_ranges(df, num_cols):
    ranges = {}
    for col in num_cols:
        if col in df:
            vals = df[col].dropna().astype(float).tolist()
            if vals:
                min_val, max_val = min(vals), max(vals)
                ranges[col] = max_val - min_val if max_val != min_val else 1.0
            else:
                ranges[col] = 1.0
        else:
            ranges[col] = 1.0
    return ranges

def gower_dist(row, query, num_cols, cat_cols, feature_ranges_dict):
    total_distance, count = 0, 0
    for col in num_cols:
        rv, qv = row.get(col), query.get(col)
        if pd.notnull(rv) and pd.notnull(qv) and col in feature_ranges_dict:
            try:
                total_distance += abs(float(rv) - float(qv)) / feature_ranges_dict[col]
                count += 1
            except TypeError:
                pass
        elif rv is None or qv is None:
            total_distance += 1
            count += 1
    for col in cat_cols:
        rv, qv = row.get(col), query.get(col)
        if rv is not None and qv is not None:
            total_distance += 0 if rv == qv else 1
            count += 1
        elif rv is None or qv is None:
            total_distance += 1
            count += 1
    return total_distance / count if count else np.nan

def retrieve_and_reuse(current_case_df, current_query, k, num_cols, cat_cols):
    current_feature_ranges = calculate_feature_ranges(current_case_df, num_cols)
    distances = current_case_df.apply(
        lambda r: gower_dist(r, current_query, num_cols, cat_cols, current_feature_ranges),
        axis=1
    )
    valid_distances = distances.dropna()
    if valid_distances.empty:
        return pd.DataFrame(), np.nan, pd.Series(dtype='float64')
    top_k_indices = valid_distances.nsmallest(min(k, len(valid_distances))).index
    top_k_cases = current_case_df.loc[top_k_indices].copy()
    top_k_cases['gower_distance'] = distances[top_k_indices]
    if not top_k_cases.empty and 'sale_price_per_sqm' in top_k_cases:
        estimated_price = int(top_k_cases["sale_price_per_sqm"].mean())
    else:
        estimated_price = np.nan
    return top_k_cases, estimated_price, distances

# === Global Constants ===
numeric_cols = ["sqm", "lot_size", "bedrooms", "bathrooms", "year_built", "floor_number", "amenity_score", "cluster_amenity_score"]
cat_cols = ["region", "property_type", "zoning", "access_to_road"]

# === Load Dataset Function ===
@st.cache_data
def load_case_base():
    case_base_list = []
    case_base_list.extend(generate_cases("apartment", 10, 1001))
    case_base_list.extend(generate_cases("non-clustered house", 10, 2001))
    case_base_list.extend(generate_cases("land", 10, 3001))
    case_base_list.extend(generate_cases("clustered house", 10, 4001))
    df = pd.DataFrame(case_base_list)
    if 'sale_price_per_sqm' in df.columns:
        sale_price_column = df.pop('sale_price_per_sqm')
        df['sale_price_per_sqm'] = sale_price_column
    return df

# === Session State Initialization ===
if 'case_df' not in st.session_state:
    st.session_state.case_df = load_case_base()

if 'current_numeric_cols' not in st.session_state:
    st.session_state.current_numeric_cols = [col for col in numeric_cols if col in st.session_state.case_df.columns]

if 'current_cat_cols' not in st.session_state:
    st.session_state.current_cat_cols = [col for col in cat_cols if col in st.session_state.case_df.columns]

case_df = st.session_state.case_df
current_numeric_cols = st.session_state.current_numeric_cols
current_cat_cols = st.session_state.current_cat_cols


# === UI Streamlit ===
st.title("CBR Property Price Estimation")

st.header("Input Query")
with st.form("query_form"):
    region = st.selectbox("Region", case_df["region"].unique())
    property_type = st.selectbox("Property Type", case_df["property_type"].unique())
    sqm = st.number_input("sqm", min_value=0, value=50)
    lot_size = st.number_input("Lot Size", min_value=0, value=100)
    bedrooms = st.number_input("Bedrooms", min_value=0, value=2)
    bathrooms = st.number_input("Bathrooms", min_value=0, value=1)
    year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2010)
    floor_number = st.number_input("Floor Number", min_value=0, value=1)
    amenity_score = st.number_input("Amenity Score", min_value=0, max_value=5, value=3)
    cluster_amenity_score = st.number_input("Cluster Amenity Score", min_value=0, max_value=5, value=0)
    zoning = st.selectbox("Zoning", ["residential", "commercial", "agricultural", None])
    access_to_road = st.selectbox("Access to Road", [True, False, None])
    submitted = st.form_submit_button("Estimate Price")

query_dict = {
    "region": region,
    "property_type": property_type,
    "sqm": sqm,
    "lot_size": lot_size,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "year_built": year_built,
    "floor_number": floor_number,
    "amenity_score": amenity_score,
    "cluster_amenity_score": cluster_amenity_score,
    "zoning": zoning if zoning != "None" else None,
    "access_to_road": access_to_road if access_to_road != "None" else None
}

if submitted:
    top_k, estimate_price, _ = retrieve_and_reuse(case_df, query_dict, k=3, num_cols=current_numeric_cols, cat_cols=current_cat_cols)
    st.subheader("Estimated Price (per sqm)")
    st.write(f"**Rp {estimate_price:,.0f}**")

    st.subheader("Top 3 Similar Cases")
    st.dataframe(top_k)

    st.markdown("---")

st.subheader("Revise & Retain")
actual_total_price = st.number_input("Masukkan Harga Total Sebenarnya (Rp)", min_value=0, step=1000000)

if st.button("Revise & Retain"):
    if actual_total_price > 0:
        actual_price_per_sqm = actual_total_price

        new_case = query_dict.copy()
        new_case['id'] = case_df['id'].max() + 1 if not case_df.empty else 1
        new_case['sale_price_per_sqm'] = actual_price_per_sqm
        case_df.loc[len(case_df)] = new_case
        st.session_state.case_df = case_df  # ⬅️ Update session state

        st.success(f"Case baru berhasil di-retain! (ID: {new_case['id']})\nHarga per sqm: Rp {actual_price_per_sqm:,.0f}")

        st.write("Case Base Setelah Retain:")
        st.dataframe(case_df.tail(5))
    else:
        st.warning("Masukkan harga aktual per sqm terlebih dahulu.")

st.markdown("---")
st.header("Distribusi Rata-Rata Harga per sqm per Property Type")

# Hitung rata-rata harga per property_type
avg_price_per_type = case_df.groupby("property_type")["sale_price_per_sqm"].mean().reset_index()

# Tampilkan Bar Chart
st.bar_chart(avg_price_per_type.set_index("property_type"))



