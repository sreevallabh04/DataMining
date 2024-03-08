# Import pandas and sklearn modules
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read the csv file into a dataframe
df = pd.read_csv("loan_data.csv")

# Separate the features and the target variable
X = df.drop("Loan_Status", axis=1) # Features
y = df["Loan_Status"] # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit a decision tree classifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dtree.predict(X_test)

# Evaluate the accuracy of the model
acc = accuracy_score(y_test, y_pred)
print("The accuracy of the decision tree classifier is:", acc)
