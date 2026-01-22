import os
import requests
from datetime import datetime, timezone
from pymongo import MongoClient

CITY = "Karachi"
COUNTRY = "Pakistan"
LAT = 24.8607
LON = 67.0011

OPEN_METEO_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    f"?latitude={LAT}&longitude={LON}"
    "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
    "&timezone=UTC"
)

def main():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI env var not set. Run: export MONGO_URI='mongodb+srv://...'")

    # Fetch raw data
    r = requests.get(OPEN_METEO_URL, timeout=30)
    r.raise_for_status()
    payload = r.json()

    times = payload["hourly"]["time"]
    hourly = payload["hourly"]

    client = MongoClient(mongo_uri)
    db = client["feature_store"]
    raw_col = db["air_quality_raw"]

    inserted = 0
    for i, t in enumerate(times):
        doc = {
            "source": "open-meteo",
            "city": CITY,
            "country": COUNTRY,
            "location": {"lat": LAT, "lon": LON},
            "timestamp": t,  # UTC ISO string
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
            {"city": CITY, "timestamp": t},
            {"$set": doc},
            upsert=True
        )
        if res.upserted_id is not None:
            inserted += 1

    print("✅ RAW ingestion complete")
    print(f"🆕 New raw docs inserted: {inserted}")
    print(f"📊 Total raw hours processed: {len(times)}")

if __name__ == "__main__":
    main()
