from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="SocialPulse | Consumer Pain-Point Intelligence",
    page_icon="📱",
    layout="wide",
)

ASPECT_MAP = {
    "All Aspects": None,
    "Software & OS Updates": "aspect_software_os",
    "Battery & Charging": "aspect_battery_charging",
    "Camera & Video": "aspect_camera_video",
    "Display & Screen": "aspect_display_screen",
    "Performance & Speed": "aspect_performance_speed",
    "Thermals & Heating": "aspect_thermals_heating",
    "Price, Warranty & Value": "aspect_price_support_value",
}

SMARTPHONE_DOMAIN_LEXICON = {
    "drain": -2.2, "draining": -2.4, "drained": -2.0,
    "overheat": -2.6, "overheating": -2.8, "throttling": -2.3, "warm": -1.2,
    "lag": -2.2, "laggy": -2.4, "stutter": -2.3, "stuttering": -2.4,
    "glitch": -2.1, "glitchy": -2.3, "bug": -1.9, "buggy": -2.3,
    "blurry": -2.0, "grainy": -1.9, "washed": -1.6,
    "scratch": -1.8, "scratched": -2.0, "tint": -1.5, "burn": -2.2,
    "crash": -2.4, "crashing": -2.5, "freeze": -2.2, "freezing": -2.4,
    "brick": -3.0, "bricked": -3.2, "bootloop": -3.0,
    "unusable": -2.8, "downgrade": -2.2, "bloated": -1.9, "bloatware": -2.1,
    "smooth": 2.0, "snappy": 2.2, "crisp": 1.9, "flawless": 2.6, "solid": 1.7,
}

BRAND_COLORS = {
    "Apple": "#1f77b4",
    "Samsung": "#ff7f0e",
    "Google Pixel": "#2ca02c",
    "OnePlus": "#d62728",
}


def load_data():
    posts_path = Path("data/processed/reddit_analyzed_posts.csv")
    aspect_path = Path("data/processed/brand_aspect_comparison.csv")
    if not posts_path.exists() or not aspect_path.exists():
        st.error("Processed data files not found.")
        st.stop()
    df_p = pd.read_csv(posts_path)
    df_a = pd.read_csv(aspect_path)

    def assign_granular_sentiment(score):
        if score <= -0.4:
            return "🔴 Strongly Negative"
        elif score <= -0.05:
            return "🟠 Mildly Negative"
        elif score < 0.05:
            return "⚪ Neutral / Mixed"
        elif score < 0.4:
            return "🟢 Mildly Positive"
        else:
            return "⭐ Strongly Positive"

    if "granular_sentiment" not in df_p.columns:
        df_p["granular_sentiment"] = df_p["sentiment_compound"].apply(assign_granular_sentiment)

    return df_p, df_a


@st.cache_resource
def load_analyzers():
    base = SentimentIntensityAnalyzer()
    tuned = SentimentIntensityAnalyzer()
    tuned.lexicon.update(SMARTPHONE_DOMAIN_LEXICON)
    return base, tuned


df, aspect_df = load_data()
base_vader, tuned_vader = load_analyzers()

available_brands = sorted(df["brand"].unique().tolist())

# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.title("📱 SocialPulse Filter")
st.sidebar.caption("Granular Consumer Intelligence")

selected_brands = st.sidebar.multiselect(
    "Select Brands",
    options=available_brands,
    default=available_brands,
)

selected_aspect_label = st.sidebar.selectbox(
    "Filter by Product Aspect",
    options=list(ASPECT_MAP.keys()),
    index=0,
)

granular_sentiment_options = [
    "🔴 Strongly Negative",
    "🟠 Mildly Negative",
    "⚪ Neutral / Mixed",
    "🟢 Mildly Positive",
    "⭐ Strongly Positive",
]

selected_granular_sentiments = st.sidebar.multiselect(
    "Granular Sentiment Category",
    options=granular_sentiment_options,
    default=granular_sentiment_options,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Advanced Categories")

all_communities = sorted(df["community_or_context"].unique().tolist())
selected_communities = st.sidebar.multiselect(
    "Filter by Subreddit Community",
    options=all_communities,
    default=all_communities,
)

min_engagement = st.sidebar.slider(
    "Minimum Upvotes / Engagement Score",
    min_value=0,
    max_value=int(df["engagement_score"].max()),
    value=0,
)

if not selected_brands:
    st.warning("Please select at least one brand in the sidebar.")
    st.stop()

filtered_df = df[
    (df["brand"].isin(selected_brands))
    & (df["granular_sentiment"].isin(selected_granular_sentiments))
    & (df["community_or_context"].isin(selected_communities))
    & (df["engagement_score"] >= min_engagement)
].copy()

aspect_col = ASPECT_MAP[selected_aspect_label]
if aspect_col:
    filtered_df = filtered_df[filtered_df[aspect_col] == 1]

# ---------------- HEADER & KPI RIBBON ----------------
st.title("📱 SocialPulse: Consumer Pain-Point Intelligence")
st.markdown(
    "**Discover What Consumers Love, Hate, and Want Improved** — "
    "Aspect-level NLP & engineering triage across Reddit smartphone communities."
)

reclassified_count = int((df["base_sentiment_label"] != df["sentiment_label"]).sum())
neg_pct = (
    (filtered_df["sentiment_compound"] < -0.05).mean() * 100
    if len(filtered_df) > 0
    else 0.0
)
req_count = int(filtered_df["is_improvement_request"].sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Discussions in View", f"{len(filtered_df):,}", f"of {len(df):,} clean posts")
col2.metric("Negative Sentiment Rate", f"{neg_pct:.1f}%", "Domain-Tuned VADER")
col3.metric("Explicit Feature Requests", f"{req_count}", "Actionable user asks")
col4.metric(
    "Domain Lexicon Corrections",
    f"{reclassified_count} posts",
    f"{(reclassified_count / len(df) * 100):.1f}% vs. Base VADER",
)

st.markdown("---")

# ---------------- NAVIGATION TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Product Triage & Brand Comparison",
    "🔍 2. TF-IDF Complaint Phrase Explorer",
    "💬 3. Voice of Customer & Feature Requests",
    "🧪 4. Live Sentiment Calibration Sandbox",
])

# ==============================================================================
# TAB 1: PRODUCT TRIAGE & BRAND COMPARISON (CHARTS START HERE)
# ==============================================================================
with tab1:
    st.subheader("Why Macro Sentiment Fails — And What Aspect Triage Reveals")
    aspect_sub = aspect_df[aspect_df["brand"].isin(selected_brands)].copy()

    if aspect_sub.empty:
        st.warning("No aspect data available for the selected brand combination.")
    else:
        c_left, c_right = st.columns(2)

        with c_left:
            fig_triage = px.bar(
                aspect_sub.sort_values("triage_priority_score", ascending=True),
                x="triage_priority_score",
                y="aspect",
                color="brand",
                barmode="group",
                orientation="h",
                color_discrete_map=BRAND_COLORS,
                title="Engineering Triage Priority Score (Volume × Negativity × Engagement)",
                labels={
                    "triage_priority_score": "Triage Priority Score",
                    "aspect": "Product Aspect",
                    "brand": "Brand",
                },
                text_auto=".2f",
            )
            fig_triage.update_layout(height=450, legend_title_text="Brand")
            st.plotly_chart(fig_triage, use_container_width=True)

        with c_right:
            fig_quadrant = px.scatter(
                aspect_sub,
                x="share_of_voice_pct",
                y="negative_rate_pct",
                size="mentions",
                color="brand",
                hover_name="aspect",
                hover_data={
                    "brand": True,
                    "mentions": True,
                    "share_of_voice_pct": ":.1f",
                    "negative_rate_pct": ":.1f",
                    "triage_priority_score": ":.2f",
                },
                size_max=45,
                color_discrete_map=BRAND_COLORS,
                title="Pain-Point Risk Matrix (Discussion Volume vs. Negative Severity)",
                labels={
                    "share_of_voice_pct": "Share of Brand Discussions (%)",
                    "negative_rate_pct": "Negative Sentiment Rate (%)",
                    "brand": "Brand",
                },
            )
            fig_quadrant.update_layout(height=450)
            st.plotly_chart(fig_quadrant, use_container_width=True)

        st.markdown("#### Full Brand × Aspect Comparison Table")
        st.dataframe(
            aspect_sub[[
                "brand",
                "aspect",
                "mentions",
                "share_of_voice_pct",
                "negative_posts",
                "negative_rate_pct",
                "mean_sentiment",
                "median_interactions",
                "triage_priority_score",
            ]].sort_values(["brand", "triage_priority_score"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True,
        )

# ==============================================================================
# TAB 2: TF-IDF COMPLAINT PHRASE EXPLORER
# ==============================================================================
with tab2:
    st.subheader("Recurring Complaint Phrases (TF-IDF Bigrams & Trigrams)")

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        tfidf_brand = st.selectbox("Select Brand for Phrase Extraction", selected_brands)
    with ctrl2:
        tfidf_aspect = st.selectbox("Filter by Product Aspect", list(ASPECT_MAP.keys()), key="tfidf_asp")
    with ctrl3:
        top_n = st.slider("Number of Top Phrases", min_value=5, max_value=20, value=12)

    sub_tfidf = df[(df["brand"] == tfidf_brand) & (df["sentiment_compound"] < -0.05)].copy()
    if ASPECT_MAP[tfidf_aspect]:
        sub_tfidf = sub_tfidf[sub_tfidf[ASPECT_MAP[tfidf_aspect]] == 1]

    if len(sub_tfidf) < 3:
        st.info(f"Not enough negative posts ({len(sub_tfidf)}) in this slice to compute multi-post TF-IDF n-grams.")
    else:
        vec = TfidfVectorizer(ngram_range=(2, 3), min_df=1, max_features=500)
        try:
            mat = vec.fit_transform(sub_tfidf["text_tfidf"].fillna(""))
            scores = np.asarray(mat.mean(axis=0)).ravel()
            terms = np.array(vec.get_feature_names_out())
            top_idx = scores.argsort()[::-1][:min(top_n, len(terms))]

            phrase_df = pd.DataFrame({
                "Complaint Phrase (N-Gram)": terms[top_idx],
                "Mean TF-IDF Score": np.round(scores[top_idx], 4),
            }).sort_values("Mean TF-IDF Score", ascending=True)

            fig_phrases = px.bar(
                phrase_df,
                x="Mean TF-IDF Score",
                y="Complaint Phrase (N-Gram)",
                orientation="h",
                title=f"Top Recurring Negative N-Grams: {tfidf_brand} — {tfidf_aspect} (n={len(sub_tfidf)})",
                color="Mean TF-IDF Score",
                color_continuous_scale="Reds",
            )
            fig_phrases.update_layout(height=460)
            st.plotly_chart(fig_phrases, use_container_width=True)
        except ValueError:
            st.info("Try selecting 'All Aspects' to view broader recurring n-grams.")

# ==============================================================================
# TAB 3: VOICE OF CUSTOMER & FEATURE REQUESTS
# ==============================================================================
with tab3:
    st.subheader("Voice of Customer (VoC) Thread Inspector")

    fcol1, fcol2 = st.columns([1, 2])
    with fcol1:
        only_requests = st.checkbox(
            "🔥 Show Only Explicit Feature / Improvement Requests",
            value=False,
        )
    with fcol2:
        search_kw = st.text_input(
            "Search Post Titles & Text (e.g., 'battery drain', 'shutter', 'CarPlay')",
            value="",
        )

    voc_df = filtered_df.copy()
    if only_requests:
        voc_df = voc_df[voc_df["is_improvement_request"] == 1]
    if search_kw.strip():
        voc_df = voc_df[voc_df["text_sentiment"].str.contains(search_kw.strip(), case=False, na=False)]

    st.write(f"Showing **{len(voc_df)}** matching Reddit discussions:")

    display_cols = [
        "brand",
        "community_or_context",
        "title",
        "granular_sentiment",
        "matched_aspects",
        "sentiment_compound",
        "engagement_score",
        "num_comments",
        "permalink",
    ]
    st.dataframe(
        voc_df[display_cols].sort_values("engagement_score", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "permalink": st.column_config.LinkColumn("Reddit Thread"),
            "sentiment_compound": st.column_config.NumberColumn("Sentiment Score", format="%.3f"),
        },
    )

# ==============================================================================
# TAB 4: LIVE SENTIMENT CALIBRATION SANDBOX
# ==============================================================================
with tab4:
    st.subheader("Model Validation: Baseline VADER vs. Domain-Tuned VADER")
    st.markdown(
        "Standard off-the-shelf sentiment models score technical hardware failures "
        "as **Neutral (0.00)**. Test our 35-term smartphone domain lexicon:"
    )

    sample_complaints = {
        "Battery & Thermal Complaint": "Ever since the latest update my battery drains 35% overnight and the phone is overheating during basic browsing.",
        "Camera & Display Defect": "The camera shutter lag makes indoor photos blurry and I noticed a green tint and micro scratch on the display.",
        "Performance Stutter": "Animations keep stuttering and lagging when switching apps, plus Android Auto keeps crashing.",
        "Positive Hardware Praise": "Battery life is super solid and the 120Hz display feels snappy, crisp, and flawless.",
    }

    preset = st.selectbox("Choose a sample post or type your own below:", list(sample_complaints.keys()))
    user_text = st.text_area("Test Input Text:", value=sample_complaints[preset], height=90)

    if user_text.strip():
        base_score = base_vader.polarity_scores(user_text)["compound"]
        tuned_score = tuned_vader.polarity_scores(user_text)["compound"]

        def get_label(score):
            if score <= -0.05:
                return "Negative 🔴"
            elif score >= 0.05:
                return "Positive 🟢"
            return "Neutral ⚪"

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("1. Standard VADER (Off-the-Shelf)", f"{base_score:.3f}", get_label(base_score))
        sc2.metric(
            "2. SocialPulse Domain-Tuned VADER",
            f"{tuned_score:.3f}",
            f"{tuned_score - base_score:+.3f} shift ({get_label(tuned_score)})",
        )

        tokens = [w.strip(".,!?():;\"'").lower() for w in user_text.split()]
        matched_domain_terms = {
            t: SMARTPHONE_DOMAIN_LEXICON[t] for t in tokens if t in SMARTPHONE_DOMAIN_LEXICON
        }
        sc3.metric("Domain Lexicon Terms Triggered", f"{len(matched_domain_terms)} terms")

        if matched_domain_terms:
            st.write("**Triggered Smartphone Lexicon Weights:**", matched_domain_terms)