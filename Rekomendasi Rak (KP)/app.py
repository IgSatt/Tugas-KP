import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


# Load the necessary data
@st.cache
def load_data():
    df = pd.read_csv("data/processed/df.csv")
    data_rak = pd.read_csv("data/raw/data_rak.csv", sep=";")
    data_rak.dropna(inplace=True)
    no_rak_mapping = data_rak.groupby("Posisi Rak")["No. Rak"].apply(list).to_dict()
    return df, no_rak_mapping


df, no_rak_mapping = load_data()


# Function to assign rack position based on clustering
def assign_rack(df, item_input, no_rak_mapping):
    # Feature Encoding for Clustering
    satuan_dummies = pd.get_dummies(df["Satuan"])
    df_encoded = pd.concat(
        [df[["Kode Kelompok Barang", "Kuantitas"]], satuan_dummies], axis=1
    )

    # Clustering
    kmeans = KMeans(n_clusters=10, random_state=42)
    df["Cluster"] = kmeans.fit_predict(df_encoded)

    # Assign Posisi Rak and No. Rak
    posisi_rak = []
    no_rak_assigned = []
    used_no_rak = set()  # Track used racks
    max_positions = 4  # Up to 4 positions per letter A-K

    for cluster in df["Cluster"].unique():
        items_in_cluster = df[df["Cluster"] == cluster]
        position_counter = 0

        for idx, row in items_in_cluster.iterrows():
            letter = chr(65 + (cluster % 11))  # A-K based on cluster index
            position = f"{letter}{(position_counter % max_positions) + 1}"

            available_no_rak = no_rak_mapping.get(position, [])
            assigned_rak = None
            for rak in available_no_rak:
                if rak not in used_no_rak:
                    assigned_rak = rak
                    used_no_rak.add(rak)
                    break

            posisi_rak.append(position)
            no_rak_assigned.append(assigned_rak)
            position_counter += 1

    df["Posisi Rak"] = posisi_rak
    df["No. Rak Assigned"] = no_rak_assigned

    # Match the user's input with the result in the clustered DataFrame
    matching_row = df[
        (df["Kode Kelompok Barang"] == item_input["Kode Kelompok Barang"])
        & (df["Kuantitas"] == item_input["Kuantitas"])
        & (df["Satuan"] == item_input["Satuan"])
    ]

    if not matching_row.empty:
        return matching_row[["Posisi Rak", "No. Rak Assigned"]].values[0]
    else:
        return "No match found"


# Streamlit UI
st.title("Item Rack Assignment System")

# User input
st.header("Input Item Details:")
kode_kelompok_barang = st.text_input("Kode Kelompok Barang")
kuantitas = st.number_input("Kuantitas", min_value=0)
satuan = st.selectbox("Satuan", df["Satuan"].unique())

# Create input dictionary
item_input = {
    "Kode Kelompok Barang": kode_kelompok_barang,
    "Kuantitas": kuantitas,
    "Satuan": satuan,
}

if st.button("Get Assigned Rack"):
    if kode_kelompok_barang and kuantitas > 0 and satuan:
        posisi_rak, no_rak_assigned = assign_rack(df, item_input, no_rak_mapping)
        st.write(f"Assigned Posisi Rak: {posisi_rak}")
        st.write(f"Assigned No. Rak: {no_rak_assigned}")
    else:
        st.write("Please fill in all fields.")

# Debug section (optional)
if st.checkbox("Show full DataFrame"):
    st.write(df)
