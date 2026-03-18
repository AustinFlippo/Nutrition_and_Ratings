# Protein and Ratings: Do High-Protein Recipes Get Higher Ratings?

**Austin Flippo, David Li**

*DSC 80 Final Project, Winter 2026*

---

## Overview

This data science project explores whether high-protein recipes receive higher ratings on Food.com. We use an FDA-based definition: high protein = ≥ 20% of the Daily Value per serving (21 CFR 101.54). A binary flag `is_high_protein` drives our hypothesis test and EDA.

---

## Introduction

Choosing a recipe often comes down to whether others liked it. Protein is a key nutrient; we wondered if recipes that deliver more protein per serving tend to get better ratings.

**The question we focused on:** Do high-protein recipes get higher ratings than low-protein ones?

We define high vs. low using the FDA nutrient content claim: **high protein = ≥ 20% DV** per serving. Low protein = &lt; 20% DV. We analyze two datasets: recipes and user interactions.

**Datasets**

| Dataset | Rows | Description |
| --- | --- | --- |
| `RAW_recipes.csv` | 83,782 | Unique recipes with name, nutrition, steps, ingredients, tags |
| `interactions.csv` | 731,927 | User ratings and reviews per recipe |

**Key columns**

| Column | Description |
| --- | --- |
| `avg_rating` | Mean user rating (1–5) per recipe |
| `protein` | Protein % of daily value per serving |
| `calories` | Calories per serving |
| `n_ingredients` | Number of ingredients |
| `minutes` | Cooking time in minutes |

After cleaning and dropping rows missing `avg_rating`, `protein`, or `calories`, we have **77,260 recipes**.

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

1. **Merge recipes and interactions** — Left merge on `id` and `recipe_id`.
2. **Treat rating 0 as missing** — 0 = "didn't rate," not true zero.
3. **Compute avg_rating** — Mean rating per recipe.
4. **Parse nutrition** — Split `nutrition` string into columns (calories, protein, etc.).
5. **Protein 0 → NaN** — 0% DV protein is implausible.
6. **Drop incomplete rows** — Drop rows missing avg_rating, protein, or calories.
7. **Create `is_high_protein`** — True if protein ≥ 20% DV (FDA "high").
8. **Create `rating_class`** — Bin avg_rating into 3 equal-width bins: low, medium, high.

**First 5 rows of cleaned table:**

| name | avg_rating | calories | protein | minutes | n_ingredients |
| --- | --- | --- | --- | --- | --- |
| 1 brownies in the world best ever | 4.0 | 138.4 | 3.0 | 40 | 9 |
| 1 in canada chocolate chip cookies | 5.0 | 595.1 | 13.0 | 45 | 11 |
| 412 broccoli casserole | 5.0 | 194.8 | 22.0 | 40 | 9 |
| millionaire pound cake | 5.0 | 878.3 | 20.0 | 120 | 7 |
| 2000 meatloaf | 5.0 | 267.0 | 29.0 | 90 | 13 |

### Univariate Analysis

<iframe src="assets/attempt2/protein-distribution-fda.html" width="800" height="500" frameborder="0"></iframe>

Protein (% DV) is concentrated in 5–40. The red line marks the FDA high threshold (20% DV).

<iframe src="assets/attempt2/avg-rating-distribution.html" width="800" height="500" frameborder="0"></iframe>

Ratings are heavily right-skewed; most recipes score 4–5. This supports binning into low/medium/high for the prediction task.

### Bivariate Analysis

<iframe src="assets/attempt2/protein-calories-scatter.html" width="800" height="500" frameborder="0"></iframe>

Protein and calories are positively associated—higher-protein recipes tend to have more calories.

<iframe src="assets/attempt2/rating-by-protein-fda.html" width="800" height="500" frameborder="0"></iframe>

Comparison of low (&lt; 20% DV) vs high (≥ 20% DV) protein recipes. The permutation test quantifies significance.

<iframe src="assets/attempt2/protein-missingness-plot.html" width="800" height="500" frameborder="0"></iframe>

When protein is missing, recipes tend to have slightly lower average rating.

### Interesting Aggregates

**By is_high_protein:**

| is_high_protein | N | Mean avg_rating |
| --- | --- | --- |
| False | 38,265 | 4.63 |
| True | 38,995 | 4.61 |

**By rating_class:**

| rating_class | Mean Protein | Mean Calories | Mean N Ingredients |
| --- | --- | --- | --- |
| low | 31.8 | 465.6 | 9.3 |
| medium | 34.5 | 446.2 | 9.3 |
| high | 34.8 | 441.2 | 9.4 |

Higher-rated recipes have slightly more protein on average.

---

## Assessment of Missingness

### MNAR Analysis

**Protein** is likely **MNAR**. We turned protein = 0 into NaN because 0% DV is implausible and often indicates "didn't fill this in." Desserts, drinks, or recipes where no one computed nutrition may be more likely missing—so missingness could depend on the *true* protein we don't observe. Extra data (e.g., recipe category) could make this MAR.

### Missingness Dependency

Permutation tests show protein missingness **depends on** avg_rating, calories, and n_ingredients (p &lt; 0.05). It **does not depend on** contributor_id (p &gt; 0.05).

---

## Hypothesis Testing

**Question:** Do high-protein recipes get higher ratings than low-protein ones?

**Groups:** Low (is_high_protein = False, protein &lt; 20% DV) vs High (is_high_protein = True, protein ≥ 20% DV).

**Null hypothesis:** Mean avg_rating is the same for low and high protein.

**Alternative hypothesis:** Mean avg_rating is higher for high protein.

**Test statistic:** mean(avg_rating | high) − mean(avg_rating | low).

**Significance level:** α = 0.05.

**Method:** One-sided permutation test (500 reps).

**Result:** Observed difference **−0.0186** (high-protein slightly lower). **P-value ≈ 1.0**. We fail to reject H0. The data do not support that high-protein recipes get higher ratings.

---

## Framing a Prediction Problem

**Task:** Multiclass classification—predict rating_class (low/medium/high).

**Response:** `rating_class` (equal-width bins of avg_rating).

**Metric:** Accuracy and F1 (macro). Accuracy measures overall correctness; F1 (macro) treats each class equally and reveals performance on minority classes (low/medium ratings).

**Features:** protein, calories, n_ingredients, minutes (all known at prediction time).

---

## Baseline Model

**Setup:** Logistic regression on `protein` and `calories` (both quantitative). Multiclass via one-vs-rest. We use `class_weight='balanced'` so the model penalizes errors on minority classes more, and `stratify=y` in the train/test split to preserve class proportions.

**Results:** Train accuracy ~93.7%, test ~93.5%; Train F1 (macro) ~0.32, Test F1 (macro) ~0.32. Nutrition alone gives useful signal for overall correctness, but **F1 (macro) is much lower than accuracy**—ratings are skewed toward "high," so the model predicts the majority class well but struggles on "low" and "medium." F1 reveals this imbalance that accuracy hides.

---

## Final Model

**Feature engineering (2 new features in addition to protein, calories):**
1. **StandardScaler on `minutes`** — Long recipes may get lower ratings (people are busy). StandardScaler puts cooking times in a comparable range (some recipes take 3+ hours).
2. **QuantileTransformer on `n_ingredients`** — Ingredient count is right-skewed. QuantileTransformer reduces outlier influence; complex recipes may be rated differently.

**Model:** RandomForestClassifier with `class_weight='balanced'`. GridSearchCV on `max_depth` (scoring=`f1_macro`). Same train/test split as baseline for direct comparison.

**Results:** Train and test accuracy on par with or better than baseline (~93–94%); F1 (macro) on par or improved. The final model adds cooking time and ingredient complexity as predictors beyond nutrition alone.

---

## Fairness Analysis

**Groups:** Quick (minutes &lt; 30) vs Long (minutes ≥ 30).

**Metric:** Accuracy per group.

**Null hypothesis:** Same accuracy for both groups.

**Alternative hypothesis:** Different accuracy.

**Test statistic:** |accuracy(quick) − accuracy(long)|. Two-sided permutation test (1000 reps), α = 0.05.

**Result:** P-value ≈ 0.06. We fail to reject H0. No strong evidence the model treats quick vs. long recipes differently.

<iframe src="assets/attempt2/fairness-permutation.html" width="800" height="500" frameborder="0"></iframe>
