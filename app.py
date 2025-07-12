import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Pizza_sales_dataset_proper.csv")
    return df

df = load_data()

# Preprocessing for market basket
orders = df.groupby("order_id")["pizza_name_id"].apply(list)
te = TransactionEncoder()
te_ary = te.fit(orders).transform(orders)
basket_df = pd.DataFrame(te_ary, columns=te.columns_)

# Apriori and rules
frequent_itemsets = apriori(basket_df, min_support=0.005, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)

rules["antecedents"] = rules["antecedents"].apply(lambda x: next(iter(x)))
rules["consequents"] = rules["consequents"].apply(lambda x: next(iter(x)))

# Mapping pizza_name_id to readable name
name_map = df.drop_duplicates("pizza_name_id")[["pizza_name_id", "pizza_name"]].set_index("pizza_name_id").to_dict()["pizza_name"]

# Streamlit UI
st.title("🍕 Pizza Recommender")

# Input from user
selected_pizza = st.selectbox("Select a pizza you like:", sorted(df["pizza_name"].unique()))
pizza_id = df[df["pizza_name"] == selected_pizza]["pizza_name_id"].iloc[0]

# Recommend pizzas based on rules
recommended = rules[
    (rules["antecedents"] == pizza_id) | (rules["consequents"] == pizza_id)
].sort_values(by="confidence", ascending=False)


if not recommended.empty:
    st.subheader("People who ordered this also ordered:")
    for i, row in recommended.iterrows():
        st.markdown(f"- **{name_map[row['consequents']]}** (Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f})")
else:
    st.info("No strong recommendations found for this pizza.")

# Show optional matching by category
if st.checkbox("Show similar pizzas by category"):
    category = df[df["pizza_name_id"] == pizza_id]["pizza_category"].iloc[0]
    st.write(f"Other {category} pizzas:")
    similar = df[(df["pizza_category"] == category) & (df["pizza_name_id"] != pizza_id)]["pizza_name"].unique()
    for name in sorted(similar):
        st.markdown(f"- {name}")

