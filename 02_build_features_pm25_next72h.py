import os
from datetime import datetime, timezone
import pandas as pd
from pymongo import MongoClient

CITY = "Karachi"

RAW_COL = "air_quality_raw"

# NEW collection (do NOT change your existing 1h collection)
FEAT_COL_72H = "air_quality_features_karachi_pm25_72h"

HORIZON_HOURS = 72  # 3 days = 72 hours


def _parse_mixed_utc_timestamps(series: pd.Series) -> pd.Series:
    """
    Accepts mixed timestamp strings:
      - "YYYY-MM-DDTHH:MM"
      - "YYYY-MM-DDTHH:MM:SS"
      - "YYYY-MM-DDTHH:MM:SSZ"
      - "YYYY-MM-DDTHH:MM:SS+00:00"
    Returns timezone-aware UTC pandas datetime, with bad values -> NaT.
    """
    ts = series.astype(str).str.strip()

    # Convert trailing "Z" -> "+00:00"
    ts = ts.str.replace(r"Z$", "+00:00", regex=True)

    # If format is "YYYY-MM-DDTHH:MM" (length 16), add seconds + timezone
    ts = ts.where(ts.str.len() != 16, ts + ":00+00:00")

    # If format is "YYYY-MM-DDTHH:MM:SS" (length 19), add timezone
    ts = ts.where(ts.str.len() != 19, ts + "+00:00")

    return pd.to_datetime(ts, utc=True, errors="coerce")


def main():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError(
            "MONGO_URI env var not set. Run: export MONGO_URI='mongodb+srv://...'"
        )

    client = MongoClient(mongo_uri)
    db = client["feature_store"]
    raw_col = db[RAW_COL]
    feat_col = db[FEAT_COL_72H]

    # 1) Load raw data (Karachi only)
    cursor = raw_col.find({"city": CITY}, {"_id": 0}).sort("timestamp", 1)
    raw_list = list(cursor)

    # Need enough data for lag(24h) + horizon(72h) + buffer
    min_needed = 24 + HORIZON_HOURS + 5
    if len(raw_list) < min_needed:
        raise RuntimeError(
            f"Not enough raw data. Need at least ~{min_needed} rows, "
            f"found {len(raw_list)}. Collect more hourly data."
        )

    # 2) Convert to DataFrame
    df = pd.json_normalize(raw_list)

    # 3) Robust timestamp parsing (mixed formats)
    df["timestamp"] = _parse_mixed_utc_timestamps(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise RuntimeError("After parsing timestamps, no valid rows remained.")

    # 4) Base pollutant columns
    base_cols = [
        "pollutants.pm2_5",
        "pollutants.pm10",
        "pollutants.co",
        "pollutants.no2",
        "pollutants.so2",
        "pollutants.o3",
    ]

    # Ensure pollutant columns are numeric (sometimes APIs/old docs can store strings)
    for c in base_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Must have pm2_5 at least
    if "pollutants.pm2_5" not in df.columns:
        raise RuntimeError("Missing pollutants.pm2_5 in raw data. Cannot build features.")

    # 5) Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Mon
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # 6) Lag features (PM2.5)
    df["pm2_5_lag_1h"] = df["pollutants.pm2_5"].shift(1)
    df["pm2_5_lag_3h"] = df["pollutants.pm2_5"].shift(3)
    df["pm2_5_lag_24h"] = df["pollutants.pm2_5"].shift(24)

    # 7) Rolling features (PM2.5)
    df["pm2_5_roll_mean_3h"] = df["pollutants.pm2_5"].rolling(3, min_periods=3).mean()
    df["pm2_5_roll_mean_24h"] = df["pollutants.pm2_5"].rolling(24, min_periods=24).mean()
    df["pm2_5_roll_std_24h"] = df["pollutants.pm2_5"].rolling(24, min_periods=24).std()

    # 8) Diff feature
    df["pm2_5_diff_1h"] = df["pollutants.pm2_5"] - df["pollutants.pm2_5"].shift(1)

    # 9) Targets: PM2.5 for the next 72 hours (t+1h ... t+72h)
    target_cols = []
    for h in range(1, HORIZON_HOURS + 1):
        col = f"target_pm2_5_t_plus_{h}h"
        df[col] = df["pollutants.pm2_5"].shift(-h)
        target_cols.append(col)

    # 10) Build output docs
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
        *target_cols,
    ]

    # Some old docs might miss some pollutant keys -> ensure columns exist
    for c in out_cols:
        if c not in df.columns:
            df[c] = pd.NA

    df_out = df[out_cols].copy()

    # Convert timestamp to consistent ISO string (UTC Z)
    df_out["timestamp"] = df_out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Metadata
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    df_out["features_built_at"] = built_at
    df_out["max_target_horizon_hours"] = HORIZON_HOURS

    # 11) Upsert by (city, timestamp)
    inserted = 0
    processed = 0

    for rec in df_out.to_dict(orient="records"):
        processed += 1
        res = feat_col.update_one(
            {"city": rec.get("city", CITY), "timestamp": rec["timestamp"]},
            {"$set": rec},
            upsert=True
        )
        if res.upserted_id is not None:
            inserted += 1

    print("✅ 72-hour feature engineering complete")
    print(f"🆕 New feature docs inserted: {inserted}")
    print(f"📊 Total feature docs processed: {processed}")
    print(f"📦 Collection: feature_store.{FEAT_COL_72H}")
    print("ℹ️ Note: last 72 rows will have NaNs in target columns (no future data yet).")


if __name__ == "__main__":
    main()
