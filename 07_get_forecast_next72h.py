#!/usr/bin/env python3
import os
import json
from pymongo import MongoClient, DESCENDING, ASCENDING

DB_NAME = "feature_store"
FORECAST_COL = "aqi_forecast_karachi_next72h"
CITY_DEFAULT = "Karachi"

def main():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI env var not set")

    city = os.getenv("CITY", CITY_DEFAULT)

    client = MongoClient(mongo_uri)
    col = client[DB_NAME][FORECAST_COL]

    # 1) Find latest base_time (ISO string sorting works correctly)
    latest_doc = col.find_one(
        {"city": city},
        sort=[("base_time", DESCENDING)]
    )

    if not latest_doc:
        print(json.dumps({
            "ok": False,
            "error": f"No forecast found for city={city}",
            "collection": f"{DB_NAME}.{FORECAST_COL}"
        }, ensure_ascii=False, indent=2))
        return

    latest_base_time = latest_doc["base_time"]

    # 2) Load all 72 rows for that base_time sorted by horizon_hours
    cursor = col.find(
        {"city": city, "base_time": latest_base_time},
        sort=[("horizon_hours", ASCENDING)]
    )

    rows = []
    for d in cursor:
        d.pop("_id", None)  # remove ObjectId for clean JSON output
        rows.append(d)

    output = {
        "ok": True,
        "city": city,
        "base_time": latest_base_time,
        "count": len(rows),
        "rows": rows
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
