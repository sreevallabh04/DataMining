# Import libraries
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
df = pd.read_csv('weather_forecast.csv')

# Define the features and the target
X = df.drop('class', axis=1) # Features are all columns except 'class'
y = df['class'] # Target is the 'class' column

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', cm)
cr = classification_report(y_test, y_pred)
print('Classification report:\n', cr)
