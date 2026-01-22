import os
import pickle
import pandas as pd
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ===============================
# MongoDB connection
# ===============================
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set")

client = MongoClient(MONGO_URI)
db = client["feature_store"]
collection = db["air_quality_features_karachi_pm25_1h"]

# ===============================
# Load data from MongoDB
# ===============================
docs = list(collection.find({}, {"_id": 0}))
df = pd.DataFrame(docs)

print(f"📦 Total rows loaded: {len(df)}")

# ===============================
# Select features + target
# ===============================
FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "pm2_5_lag_1h",
    "pm2_5_lag_3h",
    "pm2_5_lag_24h",
    "pm2_5_roll_mean_3h",
    "pm2_5_roll_mean_24h",
    "pm2_5_roll_std_24h",
]

TARGET_COL = "target_pm2_5_next_1h"

df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"🧹 Rows after NaN removal: {len(df)}")

# ===============================
# Sort by time (CRITICAL)
# ===============================
df = df.sort_values("timestamp")

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# ===============================
# Time-based train/test split
# ===============================
split_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"📚 Train rows: {len(X_train)}")
print(f"🧪 Test rows: {len(X_test)}")

# ===============================
# Train Random Forest
# ===============================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# Evaluate model
# ===============================
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print(f"📉 MAE (PM2.5 next 1h): {mae:.2f}")

# ===============================
# Save model
# ===============================
os.makedirs("models", exist_ok=True)
with open("models/rf_pm25_next1h.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved: models/rf_pm25_next1h.pkl")
