import os
import requests
from datetime import datetime, timezone
from pymongo import MongoClient

CITY = "Karachi"
COUNTRY = "Pakistan"
LAT = 24.8607
LON = 67.0011

# We explicitly ask Open-Meteo for recent history and avoid forecast storage
# (Even if API returns forecast, we will filter by "now_utc_hour")
OPEN_METEO_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    f"?latitude={LAT}&longitude={LON}"
    "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
    "&timezone=UTC"
    "&past_days=7"
    "&forecast_days=0"
)

def main():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI env var not set")

    # Now hour (UTC) -> do NOT store timestamps after this
    now_utc_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    r = requests.get(OPEN_METEO_URL, timeout=30)
    r.raise_for_status()
    payload = r.json()

    times = payload["hourly"]["time"]
    hourly = payload["hourly"]

    client = MongoClient(mongo_uri)
    db = client["feature_store"]
    raw_col = db["air_quality_raw"]

    inserted = 0
    processed = 0
    skipped_future = 0

    for i, t in enumerate(times):
        processed += 1

        # Parse ISO-like timestamp safely
        # Open-Meteo usually gives "YYYY-MM-DDTHH:MM"
        ts_dt = datetime.fromisoformat(t.replace("Z", ""))
        ts_dt = ts_dt.replace(tzinfo=timezone.utc)

        # Skip future
        if ts_dt > now_utc_hour:
            skipped_future += 1
            continue

        # Store as consistent ISO with Z
        ts_str = ts_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        doc = {
            "source": "open-meteo",
            "city": CITY,
            "country": COUNTRY,
            "location": {"lat": LAT, "lon": LON},
            "timestamp": ts_str,
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

        res = raw_col.update_one(
            {"city": CITY, "timestamp": ts_str},
            {"$set": doc},
            upsert=True
        )
        if res.upserted_id is not None:
            inserted += 1

    print("✅ RAW ingestion complete (history-only)")
    print(f"📊 Total rows from API: {len(times)} | processed loop: {processed}")
    print(f"⏭️ Skipped future rows: {skipped_future}")
    print(f"🆕 New raw docs inserted: {inserted}")
    print(f"🕒 now_utc_hour used: {now_utc_hour.isoformat()}")

if __name__ == "__main__":
    main()
