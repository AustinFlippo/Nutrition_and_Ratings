# Nutrition and Ratings: Exploring the Link Between Recipe Nutrition and User Ratings

**Austin Flippo, David Li**

*DSC 80 Final Project, Winter 2026*

---

## Introduction

Food.com (formerly Recipe1M) provided thousands of recipes with nutrition facts and user ratings—so we could ask whether what’s in a recipe predicts how people rate it.

**The question we focused on:** Can we predict whether a recipe will land in the low, medium, or high rating tier using its nutritional features?

We ended up with **77,260 recipes** after dropping rows with missing ratings or key nutrition values. The columns we used: **avg_rating** (mean user rating 1–5), **protein** (% Daily Value), **calories** (per serving), **n_ingredients** (number of ingredients), and **minutes** (cooking time).

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

Recipes and interactions lived in separate files, so we merged them to get one `avg_rating` per recipe. Any rating of 0 went to NaN—those look more like “didn’t rate” than “truly zero.” The `nutrition` column was a stringified list; we parsed it into separate columns (calories, fat, sugar, sodium, protein, etc.). Protein values of 0 also became NaN, since 0% daily value is unrealistic for most dishes and likely means bad or missing data.

**First 5 rows of the cleaned table:**

```
name,avg_rating,calories,protein,minutes,n_ingredients
1 brownies in the world    best ever,4.0,138.4,3.0,40,9
1 in canada chocolate chip cookies,5.0,595.1,13.0,45,11
412 broccoli casserole,5.0,194.8,22.0,40,9
millionaire pound cake,5.0,878.3,20.0,120,7
2000 meatloaf,5.0,267.0,29.0,90,13
```

### Univariate Analysis

<iframe
  src="assets/protein-distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Protein (% DV) falls in the 5–40 range with a right tail. We dropped recipes with protein &gt; 100 for the plot—those are odd and could be typos or very niche recipes.

### Bivariate Analysis

<iframe
  src="assets/protein-missingness-plot.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

When protein is missing, recipes tend to skew slightly lower on average rating than when it’s present. The overlap is still heavy, though, which fits the idea that protein missingness is tied to other observed variables (MAR) rather than purely random.

### Interesting Aggregates

Grouping by rating class (low / medium / high) and taking means:

| rating_class | Mean Protein | Mean Calories | Mean N Ingredients |
| ------------ | ------------ | ------------- | ------------------ |
| low | 31.8 | 465.6 | 9.3 |
| medium | 34.5 | 446.2 | 9.3 |
| high | 34.8 | 441.2 | 9.4 |

Higher rated recipes carry a bit more protein on average; calories and ingredient count stay fairly flat across groups. The gap isn’t huge, but it supports the idea that nutrition and ratings are related.

---

## Assessment of Missingness

### MNAR Analysis

**Protein** stands out as likely MNAR. We turned 0s into NaN under the assumption that 0 often means “didn’t fill this in” rather than literally zero. Desserts, drinks, or recipes where nobody bothered to compute nutrition would be more likely to end up with missing protein—so the chance of missingness depends on the *true* protein level we can’t observe. Extra info (e.g., recipe category or whether someone used a nutrition calculator) could turn this into MAR.

### Missingness Dependency

Permutation tests checked whether protein missingness depends on other columns. Missingness depends on avg_rating, calories, and n_ingredients (all p &lt; 0.05).
It does not depend on contributor_id (p &gt; 0.05). The plot above shows the avg_rating comparison.

---

## Hypothesis Testing

**Question:** Do high-protein recipes get higher ratings than low-protein ones?

We tested H0: no difference in average rating between high- and low-protein recipes, vs H1: high-protein recipes have higher ratings. High/low split at the median. Test statistic: mean(avg_rating | high protein) − mean(avg_rating | low protein)—directly measures the rating gap. α = 0.05 (standard).

One-sided permutation test (500 reps). Observed difference: **−0.0186** (high actually *slightly* lower than low). P-value: **1.0**. We fail to reject H0.

---

## Framing a Prediction Problem

**Task:** Multiclass classification—predict whether a recipe’s average rating falls in the low, medium, or high bin.

We binned `avg_rating` into three equal-width bins for `rating_class` (broad tiers, not exact scores). Used accuracy since classes are fairly balanced. Features: protein, calories, n_ingredients, minutes—all known before any ratings exist.

---

## Baseline Model

**Setup:** Logistic regression with `protein` and `calories` in an sklearn `Pipeline`. Both are quantitative, so no extra encoding.

**Results:** Train accuracy 93.66%, test 93.50%. Train and test are close, so the model generalizes. Nutrition alone gives useful signal for rating tier.

---

## Final Model

**New features:** `n_ingredients` and `minutes`. Recipe complexity and cook time plausibly affect ratings (e.g., quick vs. elaborate dishes). We scaled everything with `StandardScaler` before training.

**Algorithm:** Random forest. We tuned `max_depth` with `GridSearchCV` over {5, 10, 15, 20, 25, 30, 35, 40} and 5-fold CV. The best `max_depth` was **5**.

**Results:** Train accuracy 93.66%, test 93.50%—matching the baseline. The task was already easy; the tree adds flexibility and robustness without hurting performance on this split.

---

## Fairness Analysis

**Question:** Does the model treat quick recipes (minutes &lt; 30) worse than longer ones (minutes ≥ 30)?

We compared accuracy for quick (&lt; 30 min) vs longer recipes. H0: same accuracy for both; H1: accuracy differs. Test statistic: |accuracy(X) − accuracy(Y)|, α = 0.05.

Two-sided permutation test (1000 reps). Quick recipes had accuracy **0.9404**, longer ones **0.9321**—gap of 0.0083 in favor of quick. P-value: **0.058**. We fail to reject H0.

<iframe
  src="assets/fairness-permutation.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

---
