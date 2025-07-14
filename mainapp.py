import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("Pizza_sales_dataset_proper.csv")
    df["pizza_item"] = df["pizza_category"] + "_" + df["pizza_size"] + "_" + df["pizza_name"]
    return df

df = load_data()

# Create matrix
def create_matrix(pizza_data):
    N = len(pizza_data["order_id"].unique())
    M = len(pizza_data["pizza_item"].unique())

    user_mapper = dict(zip(np.unique(pizza_data["order_id"]), list(range(N))))
    item_mapper = dict(zip(np.unique(pizza_data["pizza_item"]), list(range(M))))
    item_inv_mapper = dict(zip(list(range(M)), np.unique(pizza_data["pizza_item"])))

    user_index = [user_mapper[i] for i in pizza_data["order_id"]]
    item_index = [item_mapper[i] for i in pizza_data["pizza_item"]]

    X = csr_matrix((pizza_data["total_price"], (item_index, user_index)), shape=(M, N))
    return X, item_mapper, item_inv_mapper

pizza_data = df[["order_id", "pizza_item", "total_price"]].copy()
X, item_mapper, item_inv_mapper = create_matrix(pizza_data)

# Title
st.title("🍕 Pizza Recommender System")
st.markdown("Get recommendations based on a pizza's name, size, and category.")

# Dropdowns
pizza_name = st.selectbox("Choose a pizza name", sorted(df["pizza_name"].unique()))
pizza_size = st.selectbox("Choose pizza size", sorted(df["pizza_size"].unique()))
pizza_category = st.selectbox("Choose pizza category", sorted(df["pizza_category"].unique()))

# Combine to pizza_item
selected_item = f"{pizza_category}_{pizza_size}_{pizza_name}"

if selected_item not in item_mapper:
    st.warning("This pizza combination was not found in the dataset.")
    st.stop()

# Recommend similar pizzas
def find_similar_pizzas(pizza_item, X, k=5):
    if pizza_item not in item_mapper:
        return []

    index = item_mapper[pizza_item]
    pizza_vec = X[index]

    if pizza_vec.nnz == 0:
        return []

    kNN = NearestNeighbors(n_neighbors=min(k + 1, X.shape[0]), algorithm="brute", metric="cosine")
    kNN.fit(X)
    pizza_vec = pizza_vec.reshape(1, -1)

    try:
        neighbors = kNN.kneighbors(pizza_vec, return_distance=False).flatten()
    except:
        return []

    similar = [item_inv_mapper[i] for i in neighbors if i != index]
    return similar[:k]

similar_pizzas = find_similar_pizzas(selected_item, X, k=5)

# Show selected pizza details
st.subheader("📌 Selected Pizza Info")
sel_df = df[df["pizza_item"] == selected_item]
sel_avg_price = sel_df["total_price"].mean()
sel_unit_price = sel_df["total_price"].sum()/ sel_df["quantity"].sum() if sel_df["quantity"].sum() > 0 else 0
st.table(pd.DataFrame({
    "Field": ["Pizza", "Size", "Category", "Avg. Price", "Unit Price"],
    "Value": [pizza_name, pizza_size, pizza_category, f"₹{sel_avg_price:.2f}", f"₹{sel_unit_price:.2f}"]
}))



# Show recommendations
st.subheader("🔁 You may also like:")
if similar_pizzas:
    rec_df = df[df["pizza_item"].isin(similar_pizzas)]
    rec_info = rec_df.groupby("pizza_item").agg({
        "pizza_name": "first",
        "pizza_size": "first",
        "pizza_category": "first",
        "total_price": "mean",
        "unit_price": "mean"
    }).reset_index()

    rec_info.rename(columns={"total_price": "Avg. Sales"}, inplace=True)
    rec_info["Avg. Sales"] = rec_info["Avg. Sales"].apply(lambda x: f"₹{x:.2f}")
    rec_info["unit_price"] = rec_info["unit_price"].apply(lambda x: f"₹{x:.2f}")

    st.dataframe(rec_info[["pizza_name", "pizza_size", "pizza_category", "Avg. Sales", "unit_price"]].rename(columns={
        "pizza_name": "Pizza Name",
        "pizza_size": "Size",
        "pizza_category": "Category",
        "unit_price": "Unit Price"
    }), use_container_width=True)
else:
    st.info("No strong recommendations found for this pizza.")

