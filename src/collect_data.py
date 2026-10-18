import time
from datetime import datetime, timezone
from pathlib import Path
import requests
import pandas as pd


class SocialPulseRedditCollector:
    """
    Collects real Apple vs. Samsung consumer pain-point discussions from Reddit
    via the Arctic Shift Public Reddit Research API (No OAuth key required).
    """

    BRAND_SUBREDDITS = {
        "Apple": ["iphone", "ios", "apple"],
        "Samsung": ["samsung", "samsunggalaxy", "GalaxyS24"],
    }

    PAIN_POINT_KEYWORDS = {
        "battery_thermals": [
            "battery", "drain", "overheat", "heating", "warm", "thermal", "charging", "sot", "battery health"
        ],
        "camera_display": [
            "camera", "blurry", "shutter", "lens", "display", "screen", "scratch", "pwm", "brightness", "tint"
        ],
        "software_bugs": [
            "bug", "glitch", "lag", "stutter", "crash", "freeze", "update", "ios", "one ui", "oneui", "notification"
        ],
        "value_complaints": [
            "issue", "problem", "disappointed", "regret", "annoying", "frustrating", "worst", "broken", "fix"
        ],
    }

    def __init__(self):
        self.records = {}
        self.headers = {
            "User-Agent": "SocialPulse-Student-Portfolio/1.0"
        }

    def _detect_pain_point_bucket(self, text: str) -> str:
        text_lower = text.lower()
        for bucket, keywords in self.PAIN_POINT_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return bucket
        return "none"

    def collect_subreddit(self, subreddit: str, brand: str, pages: int = 5):
        url = "https://arctic-shift.photon-reddit.com/api/posts/search"
        before_ts = None
        added = 0

        for _ in range(pages):
            params = {
                "subreddit": subreddit,
                "limit": 100,
                "sort": "desc",
            }
            if before_ts:
                params["before"] = before_ts

            res = requests.get(url, params=params, headers=self.headers, timeout=20)
            if res.status_code != 200:
                print(f"  [!] HTTP {res.status_code} on r/{subreddit}")
                break

            posts = res.json().get("data", [])
            if not posts:
                break

            for post in posts:
                post_id = post.get("id")
                title = (post.get("title") or "").strip()
                selftext = (post.get("selftext") or "").strip()
                if selftext in ("[removed]", "[deleted]"):
                    selftext = ""

                # Require actual discussion text, not just a link or 3-word title
                full_text = f"{title}. {selftext}".strip()
                if len(full_text) < 50:
                    continue

                bucket = self._detect_pain_point_bucket(full_text)
                if bucket == "none":
                    continue

                record_id = f"rd_{post_id}"
                if record_id not in self.records:
                    created_ts = post.get("created_utc", 0)
                    self.records[record_id] = {
                        "record_id": record_id,
                        "source": "reddit",
                        "brand": brand,
                        "community_or_context": f"r/{subreddit}",
                        "initial_query_bucket": bucket,
                        "created_utc": datetime.fromtimestamp(created_ts, tz=timezone.utc).isoformat(),
                        "title": title,
                        "selftext": selftext,
                        "text": full_text,
                        "engagement_score": int(post.get("score", 1)),
                        "upvote_ratio": float(post.get("upvote_ratio", 1.0)),
                        "num_comments": int(post.get("num_comments", 0)),
                        "flair": post.get("link_flair_text") or "Unflaired",
                        "permalink": f"https://www.reddit.com{post.get('permalink', '')}",
                    }
                    added += 1

            before_ts = posts[-1].get("created_utc")
            time.sleep(0.5)

        print(f"  [r/{subreddit:<14}] -> +{added:>3} pain-point posts collected (Total unique: {len(self.records)})")

    def run(self) -> pd.DataFrame:
        for brand, subreddits in self.BRAND_SUBREDDITS.items():
            print(f"\nCollecting {brand} discussions from Reddit...")
            for sub in subreddits:
                self.collect_subreddit(subreddit=sub, brand=brand, pages=5)
        return pd.DataFrame(list(self.records.values()))


if __name__ == "__main__":
    out_file = Path("data/raw/reddit_raw_posts.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    collector = SocialPulseRedditCollector()
    df = collector.run()

    if not df.empty:
        df.to_csv(out_file, index=False)
        print(f"\nDONE! Saved {len(df)} Reddit posts to: {out_file}")
        print("\nBrand Breakdown:")
        print(df["brand"].value_counts())
        print("\nPain-Point Category Breakdown:")
        print(df["initial_query_bucket"].value_counts())