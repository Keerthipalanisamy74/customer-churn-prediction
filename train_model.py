import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Remove customerID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to number
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove missing values
df.dropna(inplace=True)

# Convert text to numbers
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy * 100)

# Save model
joblib.dump(model, "models/churn_model.pkl")

# Save column names
joblib.dump(X.columns.tolist(), "models/columns.pkl")

print("Model saved successfully!")