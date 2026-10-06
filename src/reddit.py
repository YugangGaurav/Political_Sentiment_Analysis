import time
from datetime import datetime
import requests
import pandas as pd

from .pipeline import analyze_text

def fetch_reddit_posts(subreddits, limit, engine):
    headers = {
        "User-Agent": "PoliticalSentimentResearch/1.0 (educational dashboard)"
    }
    rows, failures = [], []
    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/new.json?limit={limit}"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                failures.append(f"r/{sub}: HTTP {response.status_code}")
                continue
            payload = response.json()
            for child in payload.get("data", {}).get("children", []):
                p = child.get("data", {})
                title = (p.get("title") or "").strip()
                body = (p.get("selftext") or "").strip()
                text = f"{title}. {body}".strip() if body else title
                if not text:
                    continue
                result = analyze_text(engine, text)
                rows.append({
                    "id": p.get("id"),
                    "subreddit": f"r/{sub}",
                    "title": title,
                    "sentiment": result["sentiment"],
                    "confidence": result["confidence"],
                    "score": int(p.get("score", 0)),
                    "num_comments": int(p.get("num_comments", 0)),
                    "timestamp": datetime.fromtimestamp(p.get("created_utc", time.time())),
                    "text": text,
                })
        except Exception as exc:
            failures.append(f"r/{sub}: {exc}")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates("id").sort_values("timestamp", ascending=False).reset_index(drop=True)
    status = "Live retrieval successful." if not failures else " | ".join(failures)
    if not rows and failures:
        status = "No live posts returned. " + status
    return df, status
