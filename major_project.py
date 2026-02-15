import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

df = pd.read_csv("diabetes.csv")
df = df.drop_duplicates()
df = df.dropna()

target_candidates = ["Outcome", "target", "diagnosis", "Class", "label"]
target_col = None
for c in target_candidates:
    if c in df.columns:
        target_col = c
        break
if target_col is None:
    target_col = df.columns[-1]

y = df[target_col]
X = df.drop(columns=[target_col])

categorical_cols = X.select_dtypes(include=["object"]).columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

numeric_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

param_grids = {
    "LogisticRegression": {"C": [0.01, 0.1, 1, 10]},
    "RandomForest": {"n_estimators": [50, 100], "max_depth": [None, 10, 20]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "KNN": {"n_neighbors": [3, 5, 7]}
}

results = {}
best_model = None
best_score = -1
best_name = ""

for name, model in models.items():
    params = param_grids.get(name)
    gs = GridSearchCV(model, params, cv=5, n_jobs=1, scoring="accuracy")
    gs.fit(X_train, y_train)
    clf = gs.best_estimator_
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "ConfusionMatrix": cm}
    if acc > best_score:
        best_score = acc
        best_model = clf
        best_name = name

joblib.dump(best_model, "best_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(encoders, "encoders.joblib")

print("Training Completed")
print("Best Model:", best_name)
print("Accuracy:", best_score)
print("Full Results:", results)

def predict_single():
    print("\nEnter values for prediction:")
    input_data = {}
    for col in X.columns:
        val = input(f"{col}: ")
        input_data[col] = float(val)
    df_in = pd.DataFrame([input_data])
    for col, le in encoders.items():
        if col in df_in.columns:
            df_in[col] = le.transform(df_in[col].astype(str))
    df_in[numeric_cols] = scaler.transform(df_in[numeric_cols])
    pred = best_model.predict(df_in)[0]
    print("Prediction:", pred)

def predict_csv():
    path = input("Enter CSV path: ")
    new_df = pd.read_csv(path)
    new_df = new_df.dropna()
    for col, le in encoders.items():
        if col in new_df.columns:
            new_df[col] = le.transform(new_df[col].astype(str))
    new_df[numeric_cols] = scaler.transform(new_df[numeric_cols])
    preds = best_model.predict(new_df)
    new_df["Prediction"] = preds
    print(new_df.head())
    new_df.to_csv("bulk_predictions.csv", index=False)
    print("Saved to bulk_predictions.csv")

while True:
    print("\n1. Predict Single Input")
    print("2. Predict from CSV")
    print("3. Exit")
    ch = input("Choose: ")
    if ch == "1":
        predict_single()
    elif ch == "2":
        predict_csv()
    else:
        break
