import os
import json
from datetime import datetime, timezone

from pymongo import MongoClient

CITY = "Karachi"
DB_NAME = "feature_store"
FORECAST_COL = "aqi_forecast_karachi_next72h"


def iso_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI not set")

    client = MongoClient(mongo_uri)
    db = client[DB_NAME]
    col = db[FORECAST_COL]

    # 1) Find latest base_time for this city
    latest = col.find_one(
        {"city": CITY},
        sort=[("base_time", -1)],
        projection={"_id": 0, "base_time": 1},
    )
    if not latest:
        raise RuntimeError(f"No forecast documents found in {DB_NAME}.{FORECAST_COL} for city={CITY}")

    base_time = latest["base_time"]

    # 2) Fetch all 72 rows for that base_time, sorted by horizon_hours
    cursor = col.find(
        {"city": CITY, "base_time": base_time},
        projection={"_id": 0},
    ).sort("horizon_hours", 1)

    rows = list(cursor)

    # Safety check
    if not rows:
        raise RuntimeError(f"Latest base_time={base_time} exists but returned 0 rows. Something is wrong.")

    # 3) Build clean JSON response
    result = {
        "city": CITY,
        "base_time": base_time,
        "count": len(rows),
        "generated_at_utc": iso_now(),
        "forecast": rows,  # each item already contains target_time, predicted_pm2_5, predicted_aqi_us, category
    }

    # Print JSON (dashboard/API will return this)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
