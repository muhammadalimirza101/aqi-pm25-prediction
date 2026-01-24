import os
import requests
from datetime import datetime, timezone
from pymongo import MongoClient

CITY = "Karachi"
COUNTRY = "Pakistan"
LAT = 24.8607
LON = 67.0011

RAW_COL_72H = "air_quality_raw_72h"

OPEN_METEO_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    f"?latitude={LAT}&longitude={LON}"
    "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
    "&timezone=UTC"
)

def main():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI env var not set")

    # Fetch data
    r = requests.get(OPEN_METEO_URL, timeout=30)
    r.raise_for_status()
    payload = r.json()

    times = payload["hourly"]["time"]
    hourly = payload["hourly"]

    # ✅ IMPORTANT: only keep times <= current UTC hour
    now_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    client = MongoClient(mongo_uri)
    db = client["feature_store"]
    raw_col = db[RAW_COL_72H]

    inserted = 0
    processed = 0
    skipped_future = 0

    for i, t in enumerate(times):
        # Open-Meteo time can be "YYYY-MM-DDTHH:MM" (no seconds, no Z)
        # Make it a proper UTC datetime
        dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)

        # Skip future hours
        if dt > now_hour:
            skipped_future += 1
            continue

        # Store timestamp in a consistent format with Z
        t_iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        doc = {
            "source": "open-meteo",
            "city": CITY,
            "country": COUNTRY,
            "location": {"lat": LAT, "lon": LON},
            "timestamp": t_iso,  # ✅ normalized
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "pollutants": {
                "pm2_5": hourly["pm2_5"][i],
                "pm10": hourly["pm10"][i],
                "co": hourly["carbon_monoxide"][i],
                "no2": hourly["nitrogen_dioxide"][i],
                "so2": hourly["sulphur_dioxide"][i],
                "o3": hourly["ozone"][i],
            },
        }

        processed += 1
        res = raw_col.update_one(
            {"city": CITY, "timestamp": t_iso},
            {"$set": doc},
            upsert=True
        )
        if res.upserted_id is not None:
            inserted += 1

    print("✅ RAW ingestion (72h pipeline) complete")
    print(f"🆕 New raw docs inserted: {inserted}")
    print(f"📊 Raw docs processed (past-only): {processed}")
    print(f"⏭️ Future hours skipped: {skipped_future}")
    print(f"📦 Collection: feature_store.{RAW_COL_72H}")
    print(f"🕒 now_hour(UTC): {now_hour.isoformat()}")

if __name__ == "__main__":
    main()
