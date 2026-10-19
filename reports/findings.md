# SocialPulse: Consumer Pain-Point Intelligence Report

## 1. Executive Summary & Sample Overview
* **Total Cleaned & Deduplicated Discussions Analyzed:** `814` Reddit self-posts
* **Brand Split:** Samsung (`397` posts) vs. Apple (`101` posts)
* **Explicit Improvement / Feature Requests Identified:** `26` posts (`3.2%` of sample)
* **Sentiment Model Validation (Day 12):** Injecting our 35-term Smartphone Hardware Defect Lexicon shifted `45` technical posts (`5.5%`) that standard VADER misclassified due to neutral-sounding hardware terms (*drain*, *shutter lag*, *stutter*, *throttling*).

---

## 2. Aspect-Level Pain-Point Comparison (Research Questions #1, #2, #4)

| brand        | aspect                  |   mentions |   share_of_voice_pct |   negative_rate_pct |   mean_sentiment |   triage_priority_score |
|:-------------|:------------------------|-----------:|---------------------:|--------------------:|-----------------:|------------------------:|
| Apple        | Software & OS Updates   |         70 |                 69.3 |                31.4 |            0.183 |                   15.1  |
| Apple        | Display & Screen        |         23 |                 22.8 |                17.4 |            0.319 |                    4.35 |
| Apple        | Camera & Video          |         10 |                  9.9 |                50   |            0.082 |                    3.43 |
| Apple        | Performance & Speed     |          9 |                  8.9 |                33.3 |            0.312 |                    2.06 |
| Apple        | Battery & Charging      |         12 |                 11.9 |                16.7 |            0.471 |                    1.37 |
| Apple        | Thermals & Heating      |          5 |                  5   |                40   |            0.138 |                    1.37 |
| Apple        | Price, Warranty & Value |          4 |                  4   |                 0   |            0.507 |                    0    |
| Google Pixel | Software & OS Updates   |         82 |                 34.9 |                50   |           -0.039 |                   24.19 |
| Google Pixel | Display & Screen        |         70 |                 29.8 |                42.9 |            0.105 |                   17.7  |
| Google Pixel | Battery & Charging      |         55 |                 23.4 |                38.2 |            0.194 |                   16.01 |
| Google Pixel | Camera & Video          |         50 |                 21.3 |                30   |            0.326 |                   10.88 |
| Google Pixel | Performance & Speed     |         22 |                  9.4 |                68.2 |           -0.223 |                   10.27 |
| Google Pixel | Price, Warranty & Value |         38 |                 16.2 |                39.5 |            0.225 |                    8.85 |
| Google Pixel | Thermals & Heating      |         11 |                  4.7 |                54.5 |            0.016 |                    3.54 |
| OnePlus      | Software & OS Updates   |         32 |                 39.5 |                46.9 |           -0.003 |                   51.91 |
| OnePlus      | Battery & Charging      |         32 |                 39.5 |                28.1 |            0.306 |                   37.22 |
| OnePlus      | Camera & Video          |         18 |                 22.2 |                50   |            0.012 |                   31.8  |
| OnePlus      | Display & Screen        |         27 |                 33.3 |                29.6 |            0.201 |                   30.53 |
| OnePlus      | Price, Warranty & Value |         17 |                 21   |                41.2 |            0.169 |                   29.95 |
| OnePlus      | Thermals & Heating      |          4 |                  4.9 |                75   |           -0.516 |                   13.97 |
| OnePlus      | Performance & Speed     |          7 |                  8.6 |                28.6 |            0.213 |                    6.14 |
| Samsung      | Software & OS Updates   |        218 |                 54.9 |                38.1 |            0.06  |                   26.19 |
| Samsung      | Battery & Charging      |        109 |                 27.5 |                38.5 |            0.171 |                   17.03 |
| Samsung      | Display & Screen        |        120 |                 30.2 |                46.7 |           -0.02  |                   15.5  |
| Samsung      | Performance & Speed     |         56 |                 14.1 |                46.4 |            0.043 |                   11.73 |
| Samsung      | Camera & Video          |         69 |                 17.4 |                42   |            0.083 |                   10.13 |
| Samsung      | Thermals & Heating      |         27 |                  6.8 |                44.4 |            0.099 |                    4.86 |
| Samsung      | Price, Warranty & Value |         35 |                  8.8 |                28.6 |            0.309 |                    4.51 |

---

## 3. Recurring Complaint Phrases via TF-IDF (Research Question #3)

### Top Apple Negative N-Grams
| brand   | complaint_phrase   |   tfidf_score |
|:--------|:-------------------|--------------:|
| Apple   | ios update         |        0.0878 |
| Apple   | ios beta           |        0.0667 |
| Apple   | using markup       |        0.0569 |
| Apple   | icloud photos      |        0.0569 |
| Apple   | screen time        |        0.0569 |
| Apple   | solve issue        |        0.0564 |
| Apple   | way fix            |        0.0485 |
| Apple   | error message      |        0.0485 |
| Apple   | how fix            |        0.0482 |
| Apple   | both devices       |        0.048  |

### Top Samsung Negative N-Grams
| brand   | complaint_phrase   |   tfidf_score |
|:--------|:-------------------|--------------:|
| Samsung | s26 ultra          |        0.0499 |
| Samsung | s25 ultra          |        0.026  |
| Samsung | after update       |        0.0222 |
| Samsung | every time         |        0.0193 |
| Samsung | battery life       |        0.0192 |
| Samsung | how fix            |        0.017  |
| Samsung | black screen       |        0.0165 |
| Samsung | home screen        |        0.015  |
| Samsung | green lines        |        0.0145 |
| Samsung | customer service   |        0.0141 |

---

## 4. Product Team Triage Recommendations (Research Question #5)

### Top 3 Priority Investigations for Apple Product Team
| aspect                |   mentions |   negative_rate_pct |   triage_priority_score |
|:----------------------|-----------:|--------------------:|------------------------:|
| Software & OS Updates |         70 |                31.4 |                   15.1  |
| Display & Screen      |         23 |                17.4 |                    4.35 |
| Camera & Video        |         10 |                50   |                    3.43 |

### Top 3 Priority Investigations for Samsung Product Team
| aspect                |   mentions |   negative_rate_pct |   triage_priority_score |
|:----------------------|-----------:|--------------------:|------------------------:|
| Software & OS Updates |        218 |                38.1 |                   26.19 |
| Battery & Charging    |        109 |                38.5 |                   17.03 |
| Display & Screen      |        120 |                46.7 |                   15.5  |

---

## 5. Methodological Limitations (For Placement Interviews)
1. **Self-Selection & Troubleshooting Bias:** Subreddits like `r/iphone`, `r/ios`, and `r/samsung` serve as peer-support forums. Negative complaint rates reflect *troubleshooting share of voice*, not overall hardware defect rates across all retail buyers.
2. **Lexicon vs. Context:** While domain-tuned VADER significantly improves sensitivity to hardware failure terms compared to vanilla VADER, sarcasm and multi-aspect contrast sentences (*"The camera is amazing, but the battery is terrible"*) assign a single document-level score across co-occurring aspects.
