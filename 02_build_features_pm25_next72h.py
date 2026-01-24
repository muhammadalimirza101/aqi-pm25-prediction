import os
from datetime import datetime, timezone
import pandas as pd
from pymongo import MongoClient

CITY = "Karachi"
RAW_COL = "air_quality_raw"
FEAT_COL_72H = "air_quality_features_karachi_pm25_72h"
HORIZON_HOURS = 72

FEATURE_COLS = [
    "hour","day_of_week","is_weekend","month",
    "pm2_5_lag_1h","pm2_5_lag_3h","pm2_5_lag_24h",
    "pm2_5_roll_mean_3h","pm2_5_roll_mean_24h","pm2_5_roll_std_24h",
    "pm2_5_diff_1h",
]

def main():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI env var not set")

    now_utc_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    client = MongoClient(mongo_uri)
    db = client["feature_store"]
    raw_col = db[RAW_COL]
    feat_col = db[FEAT_COL_72H]

    raw_list = list(raw_col.find({"city": CITY}, {"_id": 0}).sort("timestamp", 1))
    if not raw_list:
        raise RuntimeError("No raw data found in air_quality_raw")

    df = pd.json_normalize(raw_list)

    # Robust parse for mixed ISO strings
    # handles: "2026-01-21T00:00", "2026-01-21T00:00:00Z", "2026-01-21T00:00:00+00:00"
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["timestamp_dt"]).sort_values("timestamp_dt")

    # IMPORTANT: drop future timestamps (this fixes your base_time becoming 26th)
    df = df[df["timestamp_dt"] <= pd.Timestamp(now_utc_hour)]

    if len(df) < (24 + HORIZON_HOURS + 5):
        raise RuntimeError(
            f"Not enough *past* raw data (<= now). Need ~{24 + HORIZON_HOURS + 5}, found {len(df)}."
        )

    # Base pollutant columns
    df["pm2_5"] = df["pollutants.pm2_5"]

    # Time features
    df["hour"] = df["timestamp_dt"].dt.hour
    df["day_of_week"] = df["timestamp_dt"].dt.dayofweek
    df["month"] = df["timestamp_dt"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Lag / Rolling / Diff
    df["pm2_5_lag_1h"] = df["pm2_5"].shift(1)
    df["pm2_5_lag_3h"] = df["pm2_5"].shift(3)
    df["pm2_5_lag_24h"] = df["pm2_5"].shift(24)

    df["pm2_5_roll_mean_3h"] = df["pm2_5"].rolling(3).mean()
    df["pm2_5_roll_mean_24h"] = df["pm2_5"].rolling(24).mean()
    df["pm2_5_roll_std_24h"] = df["pm2_5"].rolling(24).std()

    df["pm2_5_diff_1h"] = df["pm2_5"] - df["pm2_5"].shift(1)

    # Targets t+1..t+72 from pm2_5 series
    target_cols = []
    for h in range(1, HORIZON_HOURS + 1):
        col = f"target_pm2_5_t_plus_{h}h"
        df[col] = df["pm2_5"].shift(-h)
        target_cols.append(col)

    # Output docs
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    df_out = pd.DataFrame({
        "city": df.get("city", CITY),
        "country": df.get("country", "Pakistan"),
        "source": df.get("source", "open-meteo"),
        "timestamp": df["timestamp_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "features_built_at": built_at,
        "max_target_horizon_hours": HORIZON_HOURS,
        "month": df["month"],
        "hour": df["hour"],
        "day_of_week": df["day_of_week"],
        "is_weekend": df["is_weekend"],
        "pm2_5_lag_1h": df["pm2_5_lag_1h"],
        "pm2_5_lag_3h": df["pm2_5_lag_3h"],
        "pm2_5_lag_24h": df["pm2_5_lag_24h"],
        "pm2_5_roll_mean_3h": df["pm2_5_roll_mean_3h"],
        "pm2_5_roll_mean_24h": df["pm2_5_roll_mean_24h"],
        "pm2_5_roll_std_24h": df["pm2_5_roll_std_24h"],
        "pm2_5_diff_1h": df["pm2_5_diff_1h"],
    })

    for col in target_cols:
        df_out[col] = df[col]

    # Upsert
    inserted = 0
    processed = 0
    for rec in df_out.to_dict("records"):
        processed += 1
        res = feat_col.update_one(
            {"city": rec["city"], "timestamp": rec["timestamp"]},
            {"$set": rec},
            upsert=True
        )
        if res.upserted_id is not None:
            inserted += 1

    latest_ts = df_out["timestamp"].iloc[-1]
    print("✅ 72h features built (past-only)")
    print(f"🆕 Inserted: {inserted} | processed: {processed}")
    print(f"🕒 latest feature timestamp (<= now): {latest_ts}")
    print(f"📦 Collection: feature_store.{FEAT_COL_72H}")

if __name__ == "__main__":
    main()
