import re
import html
from pathlib import Path
import pandas as pd


class SocialPulsePreprocessor:
    """
    Cleans raw Reddit posts, removes cross-subreddit duplicates,
    and extracts multi-label product aspects and consumer intent.
    """

    ASPECT_LEXICON = {
        "battery_charging": [
            r"\bbattery\b", r"\bdrain\w*\b", r"\bsot\b", r"\bscreen on time\b",
            r"\bcharg\w*\b", r"\bmah\b", r"\bpower bank\b", r"\bstandby\b"
        ],
        "thermals_heating": [
            r"\boverheat\w*\b", r"\bheat\w*\b", r"\bwarm\b", r"\bhot\b",
            r"\bthermal\w*\b", r"\bthrottl\w*\b", r"\bcooling\b"
        ],
        "camera_video": [
            r"\bcamera\w*\b", r"\bphoto\w*\b", r"\bshutter\b", r"\bblur\w*\b",
            r"\blens\w*\b", r"\bzoom\b", r"\bvideo\b", r"\bselfie\b", r"\bpost processing\b"
        ],
        "display_screen": [
            r"\bscreen\b", r"\bdisplay\b", r"\bpwm\b", r"\bbrightness\b",
            r"\bscratch\w*\b", r"\boled\b", r"\bamloed\b", r"\brefresh rate\b",
            r"\b120hz\b", r"\bgreen line\b", r"\btint\b", r"\bburn in\b"
        ],
        "software_os": [
            r"\bios\b", r"\bone ui\b", r"\boneui\b", r"\bupdate\w*\b",
            r"\bbug\w*\b", r"\bglitch\w*\b", r"\bnotification\w*\b", r"\bwidget\w*\b",
            r"\bkeyboard\b", r"\bsiri\b", r"\bbixby\b", r"\bcarplay\b", r"\bandroid auto\b"
        ],
        "performance_speed": [
            r"\blag\w*\b", r"\bstutter\w*\b", r"\bfreez\w*\b", r"\bcrash\w*\b",
            r"\bslow\w*\b", r"\bram\b", r"\bchip\b", r"\bsnapdragon\b", r"\bexynos\b", r"\bframe drop\w*\b"
        ],
        "price_support_value": [
            r"\bprice\b", r"\bexpensive\b", r"\bworth\b", r"\btrade in\b",
            r"\bwarranty\b", r"\bservice center\b", r"\bapplecare\b", r"\brepair\b", r"\bregret\b"
        ],
    }

    FEATURE_REQUEST_PATTERNS = [
        r"\bi wish\b", r"\bshould add\b", r"\bbring back\b", r"\bfeature request\b",
        r"\bwhy can't\b", r"\bwould love if\b", r"\bneeds to add\b", r"\bmissing feature\b"
    ]

    STOPWORDS = {
        "the", "and", "to", "of", "in", "is", "it", "for", "that", "on", "my", "with",
        "this", "have", "but", "be", "at", "are", "as", "was", "so", "if", "out", "not",
        "or", "from", "just", "like", "has", "an", "by", "about", "up", "can", "all",
        "get", "do", "had", "when", "what", "would", "there", "their", "which", "they",
        "one", "me", "he", "she", "we", "you", "your", "im", "ive", "dont", "cant",
        "anyone", "else", "phone", "iphone", "apple", "samsung", "galaxy", "know", "even",
        "also", "really", "still", "much", "some", "any", "only", "other", "into", "than"
    }

    def clean_for_sentiment(self, text: str) -> str:
        """Clean URLs and markdown while keeping punctuation/casing for sentiment analysis."""
        if not isinstance(text, str):
            return ""
        text = html.unescape(text)
        # Convert markdown links [label](url) -> label
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        # Remove raw URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        # Remove Reddit user/subreddit mentions
        text = re.sub(r"\b/?[ur]/[A-Za-z0-9_-]+\b", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def clean_for_tfidf(self, text: str) -> str:
        """Lowercase, strip non-alphanumeric tokens, and remove common/brand stopwords for TF-IDF."""
        text = self.clean_for_sentiment(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = [
            tok for tok in text.split()
            if len(tok) > 2 and tok not in self.STOPWORDS and not tok.isdigit()
        ]
        return " ".join(tokens)

    def extract_aspects(self, text: str) -> dict:
        """Return binary flags for each aspect and a comma-separated list of matched aspects."""
        text_lower = text.lower()
        result = {}
        matched = []

        for aspect, patterns in self.ASPECT_LEXICON.items():
            is_match = any(re.search(pat, text_lower) for pat in patterns)
            result[f"aspect_{aspect}"] = int(is_match)
            if is_match:
                matched.append(aspect)

        result["matched_aspects"] = ",".join(matched) if matched else "general_experience"
        result["aspect_count"] = len(matched)
        return result

    def detect_feature_request(self, text: str) -> int:
        """Flag posts that explicitly request a missing or improved feature."""
        text_lower = text.lower()
        return int(any(re.search(pat, text_lower) for pat in self.FEATURE_REQUEST_PATTERNS))

    def run(self, raw_path: Path, output_path: Path) -> pd.DataFrame:
        df = pd.read_csv(raw_path)
        initial_count = len(df)

        # 1. Drop empty text and exact record_id duplicates
        df = df.dropna(subset=["text", "title"]).drop_duplicates(subset=["record_id"])

        # 2. Deduplicate cross-posts (same user posting identical title in r/iphone and r/ios)
        df["norm_title"] = df["title"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
        df = df.drop_duplicates(subset=["brand", "norm_title"]).drop(columns=["norm_title"])

        # 3. Parse and validate UTC timestamps
        df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["created_utc"]).sort_values("created_utc", ascending=False).reset_index(drop=True)
        df["post_date"] = df["created_utc"].dt.date

        # 4. Create dual cleaned text columns
        df["text_sentiment"] = df["text"].apply(self.clean_for_sentiment)
        df["text_tfidf"] = df["text"].apply(self.clean_for_tfidf)
        df["word_count"] = df["text_sentiment"].str.split().str.len()

        # Filter out overly short posts after URL stripping
        df = df[df["word_count"] >= 10].reset_index(drop=True)

        # 5. Extract Multi-Label Product Aspects & Feature Request Intent
        aspect_df = pd.DataFrame(df["text_sentiment"].apply(self.extract_aspects).tolist())
        df = pd.concat([df, aspect_df], axis=1)
        df["is_feature_request"] = df["text_sentiment"].apply(self.detect_feature_request)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"Preprocessing Complete!")
        print(f"  Raw records:              {initial_count}")
        print(f"  Clean unique discussions: {len(df)} (Removed {initial_count - len(df)} cross-posts/short records)")
        print("\nClean Discussions by Brand:")
        print(df["brand"].value_counts())

        aspect_cols = [c for c in df.columns if c.startswith("aspect_") and c != "aspect_count"]
        print("\nProduct Aspect Mention Counts Across All Posts:")
        aspect_totals = df[aspect_cols].sum().sort_values(ascending=False)
        for col, count in aspect_totals.items():
            pct = (count / len(df)) * 100
            print(f"  {col.replace('aspect_', ''):<22}: {count:>4} posts ({pct:.1f}%)")

        print(f"\nExplicit Feature Requests detected: {df['is_feature_request'].sum()} posts")
        print(f"Saved clean dataset to: {output_path}")
        return df


if __name__ == "__main__":
    raw_csv = Path("data/raw/reddit_raw_posts.csv")
    clean_csv = Path("data/processed/reddit_clean_posts.csv")

    preprocessor = SocialPulsePreprocessor()
    preprocessor.run(raw_csv, clean_csv)