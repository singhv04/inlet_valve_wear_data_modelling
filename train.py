# train_xgboost_inlet.py

import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from xgboost import plot_importance

# ---------------------------------------
# 1. Load Dataset
# ---------------------------------------
df = pd.read_csv("data/synthetic_inlet_valve_data.csv")
print(df['wear'].describe())
print(df['wear'].value_counts().head())

# ---------------------------------------
# 2. Preprocessing
# ---------------------------------------
target = "wear"
features = [col for col in df.columns if col != target]

numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_features.remove("wear")
categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

# ---------------------------------------
# 3. Stratified Splitting (based on rpm bins)
# ---------------------------------------
df["rpm_bin"] = pd.cut(df["rpm"], bins=[0, 1200, 1500, 1800, 2100, 2500], labels=False)
df = df.dropna(subset=["rpm_bin"])
df["rpm_bin"] = df["rpm_bin"].astype(int)

train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df["rpm_bin"], random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.125, stratify=train_val_df["rpm_bin"], random_state=42)

for d in [train_df, val_df, test_df]:
    d.drop(columns=["rpm_bin"], inplace=True)

# ---------------------------------------
# 4. Preprocessing Pipelines
# ---------------------------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# ---------------------------------------
# 5. Model Pipeline: XGBoost
# ---------------------------------------
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
        n_jobs=-1
    ))
])

start = time.time()
model.fit(train_df[features], train_df[target])
print(f"Training time: {time.time() - start:.2f} sec")

# ---------------------------------------
# 6. Evaluation Function
# ---------------------------------------
def evaluate(name, X, y_true):
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    print(f"[{name}] R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")

    return y_true, y_pred

print("\nModel Evaluation:")
y_val, y_pred_val = evaluate("Validation", val_df[features], val_df[target])
y_test, y_pred_test = evaluate("Test", test_df[features], test_df[target])

# ---------------------------------------
# 7. Residual Plot
# ---------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_val - y_pred_val, alpha=0.3, label="Validation", color="blue")
plt.scatter(y_test, y_test - y_pred_test, alpha=0.3, label="Test", color="green")
plt.axhline(0, linestyle='--', color='black')
plt.title("XGBoost Residual Plot")
plt.xlabel("True Wear")
plt.ylabel("Residuals")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/xgboost_residuals.png")
plt.close()

# ---------------------------------------
# 8. Save Model
# ---------------------------------------
joblib.dump(model, "results/inlet_xgb_model.joblib")
print("\n✅ Model saved as results/inlet_xgb_model.joblib")



# XGBoost model with raw feature names preserved
feature_names = train_df[features].columns.tolist()

plt.figure(figsize=(10, 6))
plot_importance(
    model.named_steps['regressor'],
    # importance_type='weight',  # or 'gain', 'cover'
    xlabel='Importance score',
    height=0.5,
    show_values=True,
    grid=True,
    max_num_features=15,
    importance_type='gain'
)
plt.title("XGBoost Feature Importance")
plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
plt.tight_layout()
plt.savefig("results/xgb_feature_importance.png")
# plt.show()