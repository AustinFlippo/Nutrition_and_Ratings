# Protein and Ratings: Do High-Protein Recipes Get Higher Ratings?

**Authors: Austin Flippo & David Li**

*DSC 80 Final Project, UCSD, Winter 2026*

---

## Overview

This data science project, conducted at UCSD, explores whether high-protein recipes receive higher ratings on Food.com. We use an FDA-based definition of "high protein" (≥ 20% of the Daily Value per serving) and investigate whether that threshold is associated with better user ratings.

---

## Introduction

Choosing a recipe often comes down to whether others liked it. Protein is one of the most discussed macronutrients, and high-protein diets have grown in popularity for fitness and health reasons. This naturally leads to a question: do recipes that deliver more protein per serving actually earn better ratings from users?

**The central question of this project:** Do high-protein recipes get higher ratings than low-protein ones?

To investigate this, we analyzed two datasets consisting of recipes and user interactions posted since 2008 on [food.com](https://www.food.com/). The first dataset, `RAW_recipes.csv`, contains 83,782 rows, each representing a unique recipe, with columns including the recipe name, tags, preparation time, ingredients, and a `nutrition` field storing calorie and nutrient information. The second dataset, `interactions.csv`, contains 731,927 rows, each representing a user's rating and review of a specific recipe.

We define "high protein" using the FDA nutrient content claim standard: **≥ 20% of the Daily Value (DV) per serving**. This gives us a meaningful, real-world threshold rather than an arbitrary cutoff. The most relevant columns for our analysis are described below.

| Column | Description |
| --- | --- |
| `avg_rating` | Mean user rating (1–5) per recipe |
| `protein` | Protein content as % of Daily Value per serving |
| `calories` | Total calories per serving |
| `n_ingredients` | Number of ingredients in the recipe |
| `minutes` | Cooking time in minutes |

After merging the datasets and performing data cleaning (described below), we retained **77,260 recipes** for analysis.

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

To prepare the data for analysis, we performed the following cleaning steps.

First, we left-merged the recipes and interactions datasets on `id` and `recipe_id`, matching each recipe to its user ratings and reviews. Next, we replaced all ratings of 0 with `NaN`, since a rating of 0 on a 1–5 scale indicates that the user did not actually submit a rating rather than a true rating of zero. From this, we computed `avg_rating` as the mean rating per recipe.

We then parsed the `nutrition` column, which stored all nutrient values as a single string, into individual numeric columns for calories, protein, fat, and so on. Because a protein value of 0% DV is implausible for any real recipe and typically indicates that the field was never filled in, we also replaced protein values of 0 with `NaN`. We then dropped all rows missing `avg_rating`, `protein`, or `calories`, as these are essential for our analysis. These steps yield a clean, analysis-ready dataset with no missing values in key columns.

Finally, we created two derived columns: `is_high_protein`, a boolean flag set to `True` when protein ≥ 20% DV (the FDA "high protein" threshold) — this is our prediction target — and `rating_class`, which bins `avg_rating` into three equal-width categories (low, medium, high) for exploratory analysis.

The first five rows of the cleaned dataset are shown below.

| name | avg_rating | calories | protein | minutes | n_ingredients |
| --- | --- | --- | --- | --- | --- |
| 1 brownies in the world best ever | 4.0 | 138.4 | 3.0 | 40 | 9 |
| 1 in canada chocolate chip cookies | 5.0 | 595.1 | 13.0 | 45 | 11 |
| 412 broccoli casserole | 5.0 | 194.8 | 22.0 | 40 | 9 |
| millionaire pound cake | 5.0 | 878.3 | 20.0 | 120 | 7 |
| 2000 meatloaf | 5.0 | 267.0 | 29.0 | 90 | 13 |

### Univariate Analysis

We first examined the distribution of protein (% DV) across all recipes. As shown below, the distribution is concentrated between 5% and 40% DV, with relatively few recipes exceeding 60%. The red vertical line marks the FDA high-protein threshold of 20% DV, which splits the dataset roughly in half.

<iframe src="assets/protein-distribution-fda.html" width="800" height="500" frameborder="0"></iframe>

We also examined the distribution of average ratings. Ratings are heavily left-skewed; most recipes score 4–5. We retain a binned `rating_class` for exploratory analysis, but our prediction task uses the FDA binary flag `is_high_protein` instead.

<iframe src="assets/avg-rating-distribution.html" width="800" height="500" frameborder="0"></iframe>

### Bivariate Analysis

To explore the relationship between protein and calories, we created a scatter plot of the two variables. Protein and calories are positively associated — recipes with more protein per serving tend to also have more total calories, which makes sense given that protein-dense foods like meat and legumes are calorie-dense as well.

<iframe src="assets/protein-calories-scatter.html" width="800" height="500" frameborder="0"></iframe>

We also compared average ratings between low-protein (< 20% DV) and high-protein (≥ 20% DV) recipes. The distributions appear nearly identical, suggesting that protein level alone may not be a strong predictor of rating — a result we formally test in the Hypothesis Testing section.

<iframe src="assets/rating-by-protein-fda.html" width="800" height="500" frameborder="0"></iframe>

### Interesting Aggregates

To further understand the relationship between protein level and rating, we computed the mean average rating broken down by `is_high_protein`.

| is_high_protein | N | Mean avg_rating |
| --- | --- | --- |
| False | 38,265 | 4.63 |
| True | 38,995 | 4.61 |

High-protein recipes actually score marginally *lower* on average, though the difference is tiny (0.02 points). We also grouped recipes by `rating_class` and computed mean protein, calories, and ingredient count per group.

| rating_class | Mean Protein | Mean Calories | Mean N Ingredients |
| --- | --- | --- | --- |
| low | 31.8 | 465.6 | 9.3 |
| medium | 34.5 | 446.2 | 9.3 |
| high | 34.8 | 441.2 | 9.4 |

Interestingly, higher-rated recipes tend to have slightly more protein and fewer calories on average. This pattern is subtle but consistent across classes.

---

## Assessment of Missingness

### NMAR Analysis

We believe the missingness of the `protein` column is likely **MNAR** (Missing Not at Random). We converted protein values of 0% DV to `NaN` because a true zero is nutritionally implausible — essentially no common food ingredient has zero protein. These zeros instead reflect cases where the contributor never filled in the nutrition information. Crucially, the type of recipe that is most likely to have nutrition left blank — desserts, drinks, or casual community recipes — may also be the type of recipe with genuinely low protein content. This means the missingness could depend on the unobserved true protein value itself, which is the definition of MNAR. Obtaining recipe category data (e.g., whether a recipe is tagged as a dessert or beverage) could potentially make this missingness MAR, as category might explain why nutrition was left blank.

### Missingness Dependency

We performed permutation tests to assess whether the missingness of `protein` depends on other columns in the dataset. We found that protein missingness **does depend** on `avg_rating`, `calories`, and `n_ingredients` (all p-values < 0.05), meaning recipes with missing protein values differ systematically from those with protein recorded on these dimensions. On the other hand, protein missingness **does not depend** on `contributor_id` (p-value > 0.05), suggesting that the missingness is not driven by any particular user's habits.

The plot below shows the distribution of average rating for recipes where protein is missing versus not missing. When protein is missing, recipes tend to have slightly lower average ratings, consistent with our finding that missingness depends on `avg_rating`.

<iframe src="assets/protein-missingness-plot.html" width="800" height="500" frameborder="0"></iframe>

---

## Hypothesis Testing

We investigated whether high-protein recipes (≥ 20% DV) receive higher ratings than low-protein recipes (< 20% DV).

**Null Hypothesis:** The mean average rating is the same for low-protein and high-protein recipes. Any observed difference is due to random chance.

**Alternative Hypothesis:** The mean average rating is higher for high-protein recipes than for low-protein recipes.

**Test Statistic:** mean(avg_rating | high protein) − mean(avg_rating | low protein). We chose a difference in means rather than an absolute difference because our hypothesis is directional — we are specifically asking whether high-protein recipes rate *higher*.

**Significance Level:** α = 0.05. This is a standard threshold that balances the risk of false positives and false negatives.

**Method:** We used a one-sided permutation test with 500 repetitions. A permutation test is appropriate here because we make no assumptions about the underlying distribution of ratings, and we want to assess whether the two groups plausibly come from the same population.

The observed difference in means was **−0.0186** — high-protein recipes actually rated slightly *lower* on average. This gives a **p-value ≈ 1.0**. We fail to reject the null hypothesis. The data do not support the claim that high-protein recipes earn higher ratings; if anything, the observed trend runs in the opposite direction, though the magnitude is negligibly small.

---

## Framing a Prediction Problem

Having explored the relationship between protein and ratings, we now ask: can we predict whether a recipe meets the FDA "high protein" threshold (≥ 20% DV) based on its other nutritional and structural features?

We frame this as a **binary classification** problem, predicting `is_high_protein` — the FDA-based flag (21 CFR 101.54) indicating whether a recipe meets the high-protein threshold. We switched to this binary target because the prior three-class rating prediction (low/medium/high) yielded very low macro F1 (~0.32) due to severe class imbalance. The FDA binary flag offers a more balanced and interpretable prediction task.

We evaluate our models using both **accuracy** and **F1 (macro)**. With balanced classes (~50/50), either metric is appropriate; we report both for completeness. We exclude `protein` from the feature set to avoid trivial prediction (since `is_high_protein` is directly defined by protein ≥ 20). All features used — `calories`, `minutes`, `n_ingredients`, and `total_fat` — are properties of the recipe itself and would be known at the time a recipe is posted, before any ratings have been submitted. This ensures there is no data leakage in our model.

---

## Baseline Model

Our baseline model is a logistic regression classifier trained on two quantitative features: `calories` (total calories per serving) and `minutes` (cooking time). Both are scaled with `StandardScaler` and combined with the model in a single sklearn `Pipeline`. We use `stratify=y` in the train/test split to preserve class proportions.

On the test set, the baseline achieved approximately **75.5% accuracy** and a **macro F1 score of ~0.75**. We do not consider this baseline "good"—it is a minimal starting point; the final model adds features to improve.

---

## Final Model

We chose the **RandomForestClassifier** as our final model. To improve upon the baseline, we added two additional features beyond `calories` and `minutes`: `n_ingredients` (recipe complexity) and `total_fat` (fat content). We chose these because they correlate with recipe type—e.g., protein-rich mains have different fat profiles than desserts. StandardScaler puts features on the same scale and avoids one feature dominating.

The RandomForestClassifier achieved a **test macro F1 of 0.7759** on the held-out test set, outperforming the logistic regression baseline. We selected this model based on its stronger F1 performance in cross-validation.

---

## Fairness Analysis

To evaluate whether our final model performs equitably across different types of recipes, we split recipes into two groups: **quick recipes** (cooking time < 30 minutes) and **long recipes** (cooking time ≥ 30 minutes). We evaluated fairness using **accuracy** as our metric.

**Null Hypothesis:** Our model is fair. Its accuracy for quick recipes and long recipes is roughly the same, and any observed difference is due to random chance.

**Alternative Hypothesis:** Our model is unfair. Its accuracy differs between quick and long recipes.

**Test Statistic:** |accuracy(quick) − accuracy(long)|. We used an absolute difference because our alternative hypothesis is two-sided — we have no prior expectation about which group the model would favor.

**Significance Level:** α = 0.05.

We ran a two-sided permutation test with 1,000 repetitions. The resulting **p-value was approximately 0.06**, which is just above our significance threshold. We therefore **fail to reject the null hypothesis** and conclude there is no strong evidence that our model treats quick and long recipes differently. The result is borderline, however, and a larger dataset or stricter threshold could yield a different conclusion.

<iframe src="assets/fairness-permutation.html" width="800" height="500" frameborder="0"></iframe>
