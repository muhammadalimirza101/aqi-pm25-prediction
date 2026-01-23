import os
import pandas as pd
from datetime import datetime, timezone
from pymongo import MongoClient

CITY = "Karachi"

def normalize_ts(ts: str):
    # Handles: "2026-01-26T23:00" and "2026-01-26T23:00:00Z"
    s = str(ts).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # If no seconds, pandas will still parse it fine
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    return dt

def main():
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise RuntimeError("MONGO_URI not set")

    now = datetime.now(timezone.utc)

    client = MongoClient(uri)
    db = client["feature_store"]
    col = db["air_quality_raw"]

    docs = list(col.find({"city": CITY}, {"_id": 1, "timestamp": 1}))
    if not docs:
        print("No raw docs found.")
        return

    df = pd.DataFrame(docs)
    df["ts_parsed"] = df["timestamp"].apply(normalize_ts)

    # drop rows that couldn't parse
    df = df.dropna(subset=["ts_parsed"])

    future_ids = df.loc[df["ts_parsed"] > pd.Timestamp(now), "_id"].tolist()

    if not future_ids:
        print("✅ No future raw docs to delete.")
        return

    res = col.delete_many({"_id": {"$in": future_ids}})
    print(f"🧹 Deleted {res.deleted_count} future raw docs from air_quality_raw.")

if __name__ == "__main__":
    main()
