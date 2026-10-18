import json
from pathlib import Path


def make_notebook(cells_spec: list) -> dict:
    cells = []
    for kind, content in cells_spec:
        lines = [line + "\n" for line in content.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        if kind == "md":
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": lines,
            })
        else:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines,
            })
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_all_notebooks():
    nb_dir = Path("notebooks")
    nb_dir.mkdir(parents=True, exist_ok=True)

    nb1 = make_notebook([
        ("md", "# 01 - Data Collection & Raw Dataset Audit\n**Project:** SocialPulse: Consumer Pain-Point Intelligence from Social Media"),
        ("code", "import pandas as pd\n\ndf_raw = pd.read_csv('../data/raw/reddit_raw_posts.csv')\nprint(f'Total Raw Posts Collected: {len(df_raw)}')\ndf_raw.head(5)"),
        ("md", "## 1. Brand Balance & Subreddit Distribution"),
        ("code", "print('--- Posts by Brand ---')\nprint(df_raw['brand'].value_counts())\nprint('\\n--- Posts by Subreddit ---')\nprint(df_raw['community_or_context'].value_counts())"),
        ("md", "## 2. Initial Pain-Point Query Bucket Distribution"),
        ("code", "df_raw.groupby(['brand', 'initial_query_bucket'])['record_id'].count().unstack(fill_value=0)"),
    ])

    nb2 = make_notebook([
        ("md", "# 02 - Data Cleaning, Deduplication & Multi-Label Aspect Tagging\n**Project:** SocialPulse"),
        ("code", "import pandas as pd\n\ndf_clean = pd.read_csv('../data/processed/reddit_clean_posts.csv')\nprint(f'Clean Unique Discussions: {len(df_clean)}')\ndf_clean[['brand', 'community_or_context', 'title', 'matched_aspects', 'word_count']].head(8)"),
        ("md", "## 1. Multi-Label Product Aspect Coverage (Research Question #1)"),
        ("code", "aspect_cols = [c for c in df_clean.columns if c.startswith('aspect_') and c != 'aspect_count']\naspect_summary = df_clean.groupby('brand')[aspect_cols].sum().T\naspect_summary.index = [idx.replace('aspect_', '') for idx in aspect_summary.index]\naspect_summary['Total'] = aspect_summary.sum(axis=1)\naspect_summary.sort_values('Total', ascending=False)"),
        ("md", "## 2. Distribution of Aspects Mentioned per Post"),
        ("code", "df_clean['aspect_count'].value_counts().sort_index()"),
    ])

    nb3 = make_notebook([
        ("md", "# 03 - Domain-Tuned Sentiment, TF-IDF Complaint Extraction & Triage Matrix\n**Project:** SocialPulse"),
        ("code", "import pandas as pd\nfrom IPython.display import Image, display\n\ndf = pd.read_csv('../data/processed/reddit_analyzed_posts.csv')\naspect_df = pd.read_csv('../data/processed/brand_aspect_comparison.csv')\nprint(f'Analyzed Records: {len(df)}')"),
        ("md", "## 1. Baseline VADER vs. Smartphone-Domain-Tuned VADER (Day 12 Validation)"),
        ("code", "reclassified = (df['base_sentiment_label'] != df['sentiment_label']).sum()\nprint(f'Posts reclassified after domain lexicon injection: {reclassified} ({reclassified/len(df)*100:.1f}%)')\npd.crosstab(df['brand'], df['sentiment_label'], normalize='index').round(3) * 100"),
        ("md", "## 2. Aspect-Level Pain-Point & Triage Matrix (Apple vs. Samsung)"),
        ("code", "aspect_df[['brand', 'aspect', 'mentions', 'share_of_voice_pct', 'negative_rate_pct', 'mean_sentiment', 'triage_priority_score']]"),
        ("md", "## 3. Explicit Consumer Feature & Improvement Requests"),
        ("code", "requests_df = df[df['is_improvement_request'] == 1][['brand', 'community_or_context', 'title', 'matched_aspects', 'sentiment_compound']]\nprint(f'Total Explicit Improvement Requests: {len(requests_df)}')\nrequests_df.head(10)"),
        ("md", "## 4. Portfolio Visualizations"),
        ("code", "display(Image('../reports/figures/01_feature_share_of_voice.png'))\ndisplay(Image('../reports/figures/02_negative_sentiment_by_feature.png'))\ndisplay(Image('../reports/figures/03_product_triage_priority.png'))"),
    ])

    (nb_dir / "01_data_collection.ipynb").write_text(json.dumps(nb1, indent=2), encoding="utf-8")
    (nb_dir / "02_data_cleaning_eda.ipynb").write_text(json.dumps(nb2, indent=2), encoding="utf-8")
    (nb_dir / "03_sentiment_and_topics.ipynb").write_text(json.dumps(nb3, indent=2), encoding="utf-8")
    print("Created all 3 Jupyter Notebooks in notebooks/")


def build_readme():
    fence = "```"
    readme_lines = [
        "# SocialPulse: Consumer Pain-Point Intelligence from Social Media",
        "**Discover What Consumers Love, Hate, and Want Improved**",
        "",
        "Rather than a generic macro sentiment-analysis notebook, **SocialPulse** is an end-to-end consumer intelligence pipeline that collects real-world Reddit product discussions, deduplicates cross-subreddit posts, tags multi-label smartphone features, applies a domain-tuned sentiment lexicon, extracts recurring complaint n-grams via TF-IDF, and computes an engineering **Triage Priority Score** across **Apple** and **Samsung**.",
        "",
        "---",
        "",
        "## 1. Key Insight: Why Macro Sentiment Fails Product Teams",
        "",
        "When evaluating **983 raw Reddit posts** (510 Samsung, 473 Apple) across `r/iphone`, `r/ios`, `r/apple`, `r/samsung`, `r/samsunggalaxy`, and `r/GalaxyS24`, macro brand-level sentiment looks nearly identical:",
        "",
        "| Brand | Negative (%) | Neutral (%) | Positive (%) |",
        "| :--- | :---: | :---: | :---: |",
        "| **Apple** | 37.7% | 15.2% | 47.0% |",
        "| **Samsung** | 39.0% | 14.0% | 47.1% |",
        "",
        "**What SocialPulse Uncovers via Aspect-Level Triage:**",
        "Once discussions are decomposed into 7 hardware/software aspects and weighted by **Mention Share x Negative Rate x Community Engagement**, the two brands reveal completely different engineering priorities:",
        "",
        "| Brand | Rank | Product Aspect | Mentions | Negative Rate (%) | Triage Priority Score |",
        "| :--- | :---: | :--- | :---: | :---: | :---: |",
        "| **Apple** | **#1** | **Software & OS Updates** | 271 | 38.7% | **26.22** |",
        "| **Apple** | **#2** | **Display & Screen** | 99 | 34.3% | **8.49** |",
        "| **Apple** | **#3** | **Battery & Charging** | 61 | **49.2%** | **7.49** |",
        "| **Samsung** | **#1** | **Battery & Charging** | 165 | 46.7% | **35.25** |",
        "| **Samsung** | **#2** | **Performance & Speed** | 105 | **47.6%** | **23.99** |",
        "| **Samsung** | **#3** | **Software & OS Updates** | 231 | 35.1% | **18.54** |",
        "",
        "- **Actionable Takeaway for Apple:** Software/iOS stability dominates total pain-point volume (`271` mentions, triage score `26.22`), while **Battery & Charging** exhibits the highest severity when mentioned (`49.2%` negative).",
        "- **Actionable Takeaway for Samsung:** Even though Software/One UI has more total mentions (`231`), **Battery & Charging** (`35.25` triage score) and **Performance & Speed** (`47.6%` negative, `23.99` triage score) rank as the #1 and #2 urgent engineering priorities due to higher negative intensity and user engagement.",
        "- **Explicit Feature & Improvement Requests:** Identified **24 concrete user improvement requests** using conversational intent patterns.",
        "",
        "---",
        "",
        "## 2. Visual Findings",
        "",
        "### Q1: Feature Share of Voice (Apple vs. Samsung)",
        "![Feature Share of Voice](reports/figures/01_feature_share_of_voice.png)",
        "",
        "### Q2: Negative Sentiment Rate by Feature",
        "![Negative Sentiment by Feature](reports/figures/02_negative_sentiment_by_feature.png)",
        "",
        "### Q5: Product Team Triage Priority Matrix",
        "![Product Triage Priority](reports/figures/03_product_triage_priority.png)",
        "",
        "*(See [`reports/findings.md`](reports/findings.md) for the full TF-IDF bigram/trigram complaint tables and complete aspect breakdown.)*",
        "",
        "---",
        "",
        "## 3. Repository Structure",
        "",
        fence + "text",
        "SocialPulse/",
        "|-- README.md",
        "|-- requirements.txt",
        "|-- .gitignore",
        "|-- notebooks/",
        "|   |-- 01_data_collection.ipynb",
        "|   |-- 02_data_cleaning_eda.ipynb",
        "|   +-- 03_sentiment_and_topics.ipynb",
        "|-- src/",
        "|   |-- collect_data.py          # Reddit data ingestion pipeline",
        "|   |-- preprocess.py            # Deduplication, dual text cleaning & multi-label aspect tagger",
        "|   |-- analyze_pain_points.py   # Domain-tuned VADER, TF-IDF n-grams & triage scoring",
        "|   +-- finalize_portfolio.py    # Notebook & documentation generator",
        "+-- reports/",
        "    |-- findings.md              # Automated executive report with full tables",
        "    +-- figures/                 # High-DPI visualizations",
        fence,
        "",
        "---",
        "",
        "## 4. How to Reproduce the Pipeline",
        "",
        fence + "bash",
        "pip install -r requirements.txt",
        "python src/collect_data.py",
        "python src/preprocess.py",
        "python src/analyze_pain_points.py",
        fence,
        "",
        "---",
        "",
        "## 5. Methodological Rigor & Limitations",
        "1. **Domain-Specific Sentiment Calibration:** Off-the-shelf VADER scores technical hardware complaints (*battery drain*, *shutter lag*, *thermal throttling*) as neutral. We injected a 35-term smartphone defect lexicon and validated the shift against baseline VADER.",
        "2. **Observational & Self-Selection Bias:** Reddit communities (`r/iphone`, `r/samsung`) act as peer-support forums. High negative rates reflect *troubleshooting share of voice* rather than retail hardware defect rates across all consumers.",
        "3. **Data Ethics & Redistribution Compliance:** Raw user text (`data/raw/` and `data/processed/`) is excluded from version control via `.gitignore` in accordance with platform redistribution policies; the full pipeline and aggregated intelligence reports are 100% reproducible.",
    ]
    Path("README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print("Updated README.md with empirical findings and embedded chart links.")


def build_config_files():
    gitignore_lines = [
        "# Credentials",
        ".env",
        "*.pem",
        "credentials.json",
        "",
        "# Raw & Processed User Content (Platform compliance)",
        "data/raw/*",
        "data/processed/*",
        "!data/raw/.gitkeep",
        "!data/processed/.gitkeep",
        "",
        "# Python & Jupyter",
        "__pycache__/",
        "*.py[cod]",
        ".ipynb_checkpoints/",
        ".venv/",
        "venv/",
    ]
    reqs_lines = [
        "requests>=2.31.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.5.0",
        "vaderSentiment>=3.3.2",
        "matplotlib>=3.9.0",
        "seaborn>=0.13.2",
        "tabulate>=0.9.0",
        "jupyter>=1.0.0",
    ]
    Path(".gitignore").write_text("\n".join(gitignore_lines), encoding="utf-8")
    Path("requirements.txt").write_text("\n".join(reqs_lines), encoding="utf-8")
    print("Updated .gitignore and requirements.txt.")


if __name__ == "__main__":
    build_all_notebooks()
    build_readme()
    build_config_files()
    print("\nALL DONE! Your SocialPulse project is 100% complete and portfolio-ready.")