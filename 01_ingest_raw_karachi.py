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

def parse_utc_hour(ts_str: str) -> datetime:
    """
    Open-Meteo returns timestamps like '2026-01-26T23:00'
    (no Z). We treat them as UTC because timezone=UTC is used.
    """
    # If it already contains timezone info, fromisoformat can handle it.
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    # If it's naive (no tzinfo), force UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def floor_to_hour(dt: datetime) -> datetime:
    """Round down to the start of the hour."""
    return dt.replace(minute=0, second=0, microsecond=0)

def main():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI env var not set. Add it in GitHub Secrets and in env when running locally.")

    # Current time in UTC, floored to hour (so comparisons match hourly timestamps)
    now_utc = floor_to_hour(datetime.now(timezone.utc))

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
    updated = 0
    skipped_future = 0

    for i, t in enumerate(times):
        t_dt = parse_utc_hour(t)

        # ✅ Skip any future hours (forecast)
        if t_dt > now_utc:
            skipped_future += 1
            continue

        doc = {
            "source": "open-meteo",
            "city": CITY,
            "country": COUNTRY,
            "location": {"lat": LAT, "lon": LON},
            "timestamp": t_dt.isoformat().replace("+00:00", "Z"),  # store consistent UTC string
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
            {"city": CITY, "timestamp": doc["timestamp"]},
            {"$set": doc},
            upsert=True
        )

        if res.upserted_id is not None:
            inserted += 1
        else:
            updated += 1

    print("✅ RAW ingestion complete (PAST/CURRENT ONLY)")
    print(f"🕒 Now (UTC hour): {now_utc.isoformat().replace('+00:00', 'Z')}")
    print(f"🆕 Inserted: {inserted}")
    print(f"🔁 Updated: {updated}")
    print(f"⏭️ Skipped future forecast hours: {skipped_future}")
    print(f"📊 Total hours received from API: {len(times)}")

if __name__ == "__main__":
    main()
