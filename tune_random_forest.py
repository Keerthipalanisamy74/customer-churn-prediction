import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------
# Load Dataset
# -----------------------

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# -----------------------
# Data Preprocessing
# -----------------------

df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df.dropna(inplace=True)

df = pd.get_dummies(df, drop_first=True)

# -----------------------
# Features & Target
# -----------------------

X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# -----------------------
# Train Test Split
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------
# Random Forest Model
# -----------------------

rf = RandomForestClassifier(random_state=42)

# -----------------------
# Parameters to Try
# -----------------------

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5, 10]
}

# -----------------------
# Grid Search
# -----------------------

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# -----------------------
# Best Model
# -----------------------

best_model = grid_search.best_estimator_

prediction = best_model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")