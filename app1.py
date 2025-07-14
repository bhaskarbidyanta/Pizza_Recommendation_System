import streamlit as st
import pandas as pd

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Pizza_sales_dataset_proper.csv")
    return df

df = load_data()

st.title("🍕 Smarter Pizza Recommender")

# Sidebar inputs
st.sidebar.header("Choose your preferences")
category = st.sidebar.selectbox("Pizza Category", sorted(df["pizza_category"].unique()))
size = st.sidebar.selectbox("Pizza Size", sorted(df["pizza_size"].unique()))

# Filter based on inputs
filtered_df = df[(df["pizza_category"] == category) & (df["pizza_size"] == size)]

# Aggregate total sales
top_pizzas = (
    filtered_df.groupby(["pizza_name", "pizza_name_id"])["quantity"]
    .sum()
    .reset_index()
    .sort_values(by="quantity", ascending=False)
    .head(5)
)

# Show top-selling pizzas
st.subheader(f"Top {category} Pizzas ({size} size):")
if not top_pizzas.empty:
    for i, row in top_pizzas.iterrows():
        st.markdown(f"**{row['pizza_name']}** – Sold: {row['quantity']} times")
else:
    st.warning("No pizzas found for the selected category and size.")

# Optional: Suggest similar pizzas from other categories
if st.checkbox("Suggest similar pizzas from other categories"):
    target_ingredients = df[df["pizza_name_id"] == top_pizzas.iloc[0]["pizza_name_id"]]["pizza_ingredients"].iloc[0]
    st.write(f"Matching by ingredients similar to: `{target_ingredients}`")

    def count_common_ingredients(x):
        try:
            return len(set(x.split(", ")).intersection(set(target_ingredients.split(", "))))
        except:
            return 0

    df["similarity_score"] = df["pizza_ingredients"].apply(count_common_ingredients)
    similar_pizzas = df[
        (df["pizza_category"] != category) & (df["pizza_size"] == size)
    ].sort_values(by="similarity_score", ascending=False)

    similar_names = similar_pizzas["pizza_name"].unique()[:5]

    for name in similar_names:
        st.markdown(f"- {name}")
