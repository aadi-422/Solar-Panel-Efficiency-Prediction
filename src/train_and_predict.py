import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# -----------------------------
# Load datasets
# -----------------------------
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
submission = pd.read_csv("data/sample_submission.csv")

# Backup test IDs
test_ids = test["id"]

# Drop ID column
train.drop(columns=["id"], inplace=True)
test.drop(columns=["id"], inplace=True)

categorical_cols = ["string_id", "error_code", "installation_type"]

for col in train.columns:
    if col not in categorical_cols + ["efficiency"]:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")

train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

train["power"] = train["voltage"] * train["current"]
test["power"] = test["voltage"] * test["current"]

for col in categorical_cols:
    le = LabelEncoder()
    combined = list(train[col].astype(str)) + list(test[col].astype(str))
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
  
X = train.drop(columns=["efficiency"])
y = train["efficiency"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(test_scaled)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "seed": 42,
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=300,
    evals=[(dval, "validation")],
    early_stopping_rounds=20,
    verbose_eval=False,
)

def custom_score(y_true, y_pred):
    return 100 * (1 - np.sqrt(mean_squared_error(y_true, y_pred)))

val_preds = model.predict(dval)
score = custom_score(y_val, val_preds)
print(f"✅ Validation Custom Score: {score:.2f}")

test_preds = model.predict(dtest)

output = pd.DataFrame({
    "id": test_ids,
    "efficiency": test_preds
})

output.to_csv("outputs/predicted_submission.csv", index=False)
print("✅ predicted_submission.csv saved in outputs/")
