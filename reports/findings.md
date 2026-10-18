# SocialPulse: Consumer Pain-Point Intelligence Report

## 1. Executive Summary & Sample Overview
* **Total Cleaned & Deduplicated Discussions Analyzed:** `920` Reddit self-posts
* **Brand Split:** Samsung (`480` posts) vs. Apple (`440` posts)
* **Explicit Improvement / Feature Requests Identified:** `24` posts (`2.6%` of sample)
* **Sentiment Model Validation (Day 12):** Injecting our 35-term Smartphone Hardware Defect Lexicon shifted `45` technical posts (`4.9%`) that standard VADER misclassified due to neutral-sounding hardware terms (*drain*, *shutter lag*, *stutter*, *throttling*).

---

## 2. Aspect-Level Pain-Point Comparison (Research Questions #1, #2, #4)

| brand   | aspect                  |   mentions |   share_of_voice_pct |   negative_rate_pct |   mean_sentiment |   triage_priority_score |
|:--------|:------------------------|-----------:|---------------------:|--------------------:|-----------------:|------------------------:|
| Apple   | Software & OS Updates   |        271 |                 61.6 |                38.7 |            0.064 |                   26.22 |
| Apple   | Display & Screen        |         99 |                 22.5 |                34.3 |            0.114 |                    8.49 |
| Apple   | Battery & Charging      |         61 |                 13.9 |                49.2 |            0.008 |                    7.49 |
| Apple   | Camera & Video          |         84 |                 19.1 |                31   |            0.222 |                    6.49 |
| Apple   | Performance & Speed     |         24 |                  5.5 |                54.2 |           -0.126 |                    4.44 |
| Apple   | Thermals & Heating      |         13 |                  3   |                61.5 |           -0.152 |                    2.52 |
| Apple   | Price, Warranty & Value |         22 |                  5   |                27.3 |            0.251 |                    2.19 |
| Samsung | Battery & Charging      |        165 |                 34.4 |                46.7 |            0.05  |                   35.25 |
| Samsung | Performance & Speed     |        105 |                 21.9 |                47.6 |            0.044 |                   23.99 |
| Samsung | Software & OS Updates   |        231 |                 48.1 |                35.1 |            0.097 |                   18.54 |
| Samsung | Display & Screen        |        128 |                 26.7 |                43.8 |            0.061 |                   16.17 |
| Samsung | Camera & Video          |        103 |                 21.5 |                39.8 |            0.101 |                   11.84 |
| Samsung | Price, Warranty & Value |         55 |                 11.5 |                38.2 |            0.185 |                   10.07 |
| Samsung | Thermals & Heating      |         35 |                  7.3 |                51.4 |            0.013 |                    9.32 |

---

## 3. Recurring Complaint Phrases via TF-IDF (Research Question #3)

### Top Apple Negative N-Grams
| brand   | complaint_phrase   |   tfidf_score |
|:--------|:-------------------|--------------:|
| Apple   | pro max            |        0.0426 |
| Apple   | ios update         |        0.0285 |
| Apple   | every time         |        0.0222 |
| Apple   | how fix            |        0.019  |
| Apple   | screen time        |        0.0157 |
| Apple   | don want           |        0.0154 |
| Apple   | doesn work         |        0.0145 |
| Apple   | works fine         |        0.0142 |
| Apple   | update ios         |        0.0141 |
| Apple   | turned off         |        0.013  |

### Top Samsung Negative N-Grams
| brand   | complaint_phrase   |   tfidf_score |
|:--------|:-------------------|--------------:|
| Samsung | s26 ultra          |        0.0266 |
| Samsung | battery life       |        0.0231 |
| Samsung | battery drain      |        0.0194 |
| Samsung | home screen        |        0.0163 |
| Samsung | s24 ultra          |        0.0157 |
| Samsung | s24 exynos         |        0.0157 |
| Samsung | s24 snapdragon     |        0.0146 |
| Samsung | every time         |        0.0141 |
| Samsung | fast charging      |        0.0139 |
| Samsung | hours sot          |        0.0137 |

---

## 4. Product Team Triage Recommendations (Research Question #5)

### Top 3 Priority Investigations for Apple Product Team
| aspect                |   mentions |   negative_rate_pct |   triage_priority_score |
|:----------------------|-----------:|--------------------:|------------------------:|
| Software & OS Updates |        271 |                38.7 |                   26.22 |
| Display & Screen      |         99 |                34.3 |                    8.49 |
| Battery & Charging    |         61 |                49.2 |                    7.49 |

### Top 3 Priority Investigations for Samsung Product Team
| aspect                |   mentions |   negative_rate_pct |   triage_priority_score |
|:----------------------|-----------:|--------------------:|------------------------:|
| Battery & Charging    |        165 |                46.7 |                   35.25 |
| Performance & Speed   |        105 |                47.6 |                   23.99 |
| Software & OS Updates |        231 |                35.1 |                   18.54 |

---

## 5. Methodological Limitations (For Placement Interviews)
1. **Self-Selection & Troubleshooting Bias:** Subreddits like `r/iphone`, `r/ios`, and `r/samsung` serve as peer-support forums. Negative complaint rates reflect *troubleshooting share of voice*, not overall hardware defect rates across all retail buyers.
2. **Lexicon vs. Context:** While domain-tuned VADER significantly improves sensitivity to hardware failure terms compared to vanilla VADER, sarcasm and multi-aspect contrast sentences (*"The camera is amazing, but the battery is terrible"*) assign a single document-level score across co-occurring aspects.
