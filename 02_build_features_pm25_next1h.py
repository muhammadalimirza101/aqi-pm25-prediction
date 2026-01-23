import os
from datetime import datetime, timezone
import pandas as pd
from pymongo import MongoClient

CITY = "Karachi"

RAW_COL = "air_quality_raw"
FEAT_COL = "air_quality_features_karachi_pm25_1h"

def main():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI env var not set. Run: export MONGO_URI='mongodb+srv://...'")

    client = MongoClient(mongo_uri)
    db = client["feature_store"]
    raw_col = db[RAW_COL]
    feat_col = db[FEAT_COL]

    # 1) Load raw data from MongoDB (Karachi only)
    cursor = raw_col.find({"city": CITY}, {"_id": 0}).sort("timestamp", 1)
    raw_list = list(cursor)
    if len(raw_list) < 30:
        raise RuntimeError("Not enough raw data. Run ingestion first or collect more hours.")

    # 2) Convert to DataFrame
    df = pd.json_normalize(raw_list)

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")

    # 3) Base columns
    # Pollutants are nested as pollutants.pm2_5 etc after json_normalize
    base_cols = [
        "pollutants.pm2_5",
        "pollutants.pm10",
        "pollutants.co",
        "pollutants.no2",
        "pollutants.so2",
        "pollutants.o3",
    ]

    # 4) Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Mon
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # 5) Lag features (PM2.5)
    df["pm2_5_lag_1h"] = df["pollutants.pm2_5"].shift(1)
    df["pm2_5_lag_3h"] = df["pollutants.pm2_5"].shift(3)
    df["pm2_5_lag_24h"] = df["pollutants.pm2_5"].shift(24)

    # 6) Rolling features (PM2.5)
    df["pm2_5_roll_mean_3h"] = df["pollutants.pm2_5"].rolling(3).mean()
    df["pm2_5_roll_mean_24h"] = df["pollutants.pm2_5"].rolling(24).mean()
    df["pm2_5_roll_std_24h"] = df["pollutants.pm2_5"].rolling(24).std()

    # 7) Diff features
    df["pm2_5_diff_1h"] = df["pollutants.pm2_5"] - df["pollutants.pm2_5"].shift(1)

    # 8) Target: PM2.5 next 1 hour
    df["target_pm2_5_next_1h"] = df["pollutants.pm2_5"].shift(-1)

    # 9) Build feature docs (keep it clean)
    out_cols = [
        "city",
        "country",
        "source",
        "timestamp",
        "location.lat",
        "location.lon",
        *base_cols,
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "pm2_5_lag_1h",
        "pm2_5_lag_3h",
        "pm2_5_lag_24h",
        "pm2_5_roll_mean_3h",
        "pm2_5_roll_mean_24h",
        "pm2_5_roll_std_24h",
        "pm2_5_diff_1h",
        "target_pm2_5_next_1h",
    ]

    df_out = df[out_cols].copy()

    # Convert timestamp back to ISO string for Mongo
    df_out["timestamp"] = df_out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Add metadata
    ingested_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    df_out["features_built_at"] = ingested_at
    df_out["target_horizon_hours"] = 1

    # 10) Upsert into feature collection by (city, timestamp)
    inserted = 0
    processed = 0
    for rec in df_out.to_dict(orient="records"):
        processed += 1
        res = feat_col.update_one(
            {"city": rec["city"], "timestamp": rec["timestamp"]},
            {"$set": rec},
            upsert=True
        )
        if res.upserted_id is not None:
            inserted += 1

    print("✅ Feature engineering complete")
    print(f"🆕 New feature docs inserted: {inserted}")
    print(f"📊 Total feature docs processed: {processed}")
    print(f"📦 Collection: feature_store.{FEAT_COL}")

    # Note: first rows will contain nulls for lags/rolling; last row target is null.

if __name__ == "__main__":
    main()
