import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset (assuming it's in the same directory as this script)
file_path = "basket.csv"
df = pd.read_csv(file_path, header=None)

# Preprocess the data
te = TransactionEncoder()
te_ary = te.fit(df).transform(df)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Generate frequent itemsets using Apriori
min_support = 0.0001  
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

# Generate association rules
min_confidence = 1 
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Display the rules
print("Association Rules:")
print(rules[["antecedents", "consequents", "support", "confidence"]])
