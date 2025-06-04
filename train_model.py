# === Import libraries ===
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# === Load the dataset ===
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# X -> features, y -> labels (0, 1, 2)

# === Split the dataset ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 80% for training, 20% for testing

# === Train the model ===
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# === Evaluate the model ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")  # Output like: 0.97

# === Save the model ===
joblib.dump(model, "iris_model.pkl")
print("Model saved to iris_model.pkl")
