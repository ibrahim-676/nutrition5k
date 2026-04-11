import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


df = pd.read_csv("nutrition5k.csv")
df.head()
df.info()
df.describe(include='all')
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.drop_duplicates()
# Convert both ingredient columns to strings
df["actual_ingredients"] = df["actual_ingredients"].astype(str)
df["predicted_ingredients"] = df["predicted_ingredients"].astype(str)

# Replace "nan" text with empty strings (optional but clean)
df["actual_ingredients"] = df["actual_ingredients"].replace("nan", "")
df["predicted_ingredients"] = df["predicted_ingredients"].replace("nan", "")
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)
numeric_cols = [
    "actual_cal", "pred_cal", "cal_error",
    "actual_fat", "pred_fat", "fat_error",
    "actual_carb", "pred_carb", "carb_error",
    "actual_prot", "pred_prot", "prot_error"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)

df.info()
df.head()
df.describe()
    



# ============================================================
#  VISUALISATION of Nutrition5K Dataset
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")

# 1. BAR CHART — Mean Error per Nutrient (MAE)
# ------------------------------------------------------------
mae_values = df[["cal_error","fat_error","carb_error","prot_error"]].mean()

plt.figure(figsize=(7,4))
mae_values.plot(kind='bar', color=["blue","green","orange","red"])
plt.title("Mean Absolute Error (MAE) per Nutrient")
plt.ylabel("Mean Error")
plt.xlabel("Nutrient")
plt.show()


# 2. LINE GRAPH — Sorted Calorie Error
# ------------------------------------------------------------
sorted_errors = df["cal_error"].sort_values().reset_index(drop=True)

plt.figure(figsize=(8,4))
plt.plot(sorted_errors, color="purple")
plt.title("Sorted Calorie Errors")
plt.xlabel("Dish Index (sorted)")
plt.ylabel("Calorie Error")
plt.show()

# 3. SCATTER PLOT — Actual vs Predicted Calories
# ------------------------------------------------------------
plt.figure(figsize=(6,6))
plt.scatter(df["actual_cal"], df["pred_cal"], alpha=0.4, color="teal")
plt.plot([0, df["actual_cal"].max()],
         [0, df["actual_cal"].max()],
         color="black")  # ideal line
plt.title("Actual vs Predicted Calories")
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.show()


# 4. CORRELATION HEATMAP 
# ------------------------------------------------------------
plt.figure(figsize=(6,5))
corr = df[["cal_error","fat_error","carb_error","prot_error"]].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix — Nutrient Errors")
plt.show()





# 5. BAR CHART — Top 10 Largest Calorie Errors (Using INGREDIENTS)

top10 = df.nlargest(10, "cal_error")[["dish_id","cal_error"]]

plt.figure(figsize=(10,6))
sns.barplot(
    data=top10,
    x="cal_error",
    y="dish_id",
    hue="dish_id",
    palette="Reds_r",
    legend=False
)
plt.title("Top 10 Dishes With Highest Calorie Error (By dish id)")
plt.xlabel("Calorie Error")
plt.ylabel("Dish ID")
plt.show()




# 6. BAR CHART — Under vs Over Predictions 

df["bias"] = df["pred_cal"] - df["actual_cal"]

df["prediction_type"] = df["bias"].apply(
    lambda x: "Overpredicted" if x > 20 else ("Underpredicted" if x < -20 else "Accurate (±20)")
)

plt.figure(figsize=(6,4))
df["prediction_type"].value_counts().plot(kind="bar", color=["red","blue","green"])
plt.title("Prediction Accuracy Categories")
plt.ylabel("Count")
plt.xlabel("Category")
plt.show()





print("\n" + "="*60)
print("EXTENDED QUANTITATIVE EVALUATION")
print("="*60)

# ------------------------------------------------------------
# 1️⃣ MAPE (Mean Absolute Percentage Error)
# ------------------------------------------------------------
print("\n📊 MAPE (Mean Absolute Percentage Error)")
print("-" * 60)

# MAPE calculation - exclude zero actual values to avoid division by zero
def calculate_mape(actual, error):
    """Calculate MAPE excluding zero actual values"""
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return (error[mask].abs() / actual[mask] * 100).mean()

mape_cal = calculate_mape(df["actual_cal"], df["cal_error"])
mape_fat = calculate_mape(df["actual_fat"], df["fat_error"])
mape_carb = calculate_mape(df["actual_carb"], df["carb_error"])
mape_prot = calculate_mape(df["actual_prot"], df["prot_error"])

print(f"Calories:    {mape_cal:.2f}%")
print(f"Fat:         {mape_fat:.2f}%")
print(f"Carbs:       {mape_carb:.2f}%")
print(f"Protein:     {mape_prot:.2f}%")

# Print how many zero values were excluded
print("\nNote: MAPE excludes dishes with zero actual values:")
print(f"  Calories with 0g:  {(df['actual_cal'] == 0).sum()} dishes excluded")
print(f"  Fat with 0g:       {(df['actual_fat'] == 0).sum()} dishes excluded")
print(f"  Carbs with 0g:     {(df['actual_carb'] == 0).sum()} dishes excluded")
print(f"  Protein with 0g:   {(df['actual_prot'] == 0).sum()} dishes excluded")

# ------------------------------------------------------------
# 2️⃣ RMSE (Root Mean Squared Error)
# ------------------------------------------------------------
print("\n📊 RMSE (Root Mean Squared Error)")
print("-" * 60)

rmse_cal = np.sqrt((df["cal_error"] ** 2).mean())
rmse_fat = np.sqrt((df["fat_error"] ** 2).mean())
rmse_carb = np.sqrt((df["carb_error"] ** 2).mean())
rmse_prot = np.sqrt((df["prot_error"] ** 2).mean())

print(f"Calories:    {rmse_cal:.2f} kcal")
print(f"Fat:         {rmse_fat:.2f} g")
print(f"Carbs:       {rmse_carb:.2f} g")
print(f"Protein:     {rmse_prot:.2f} g")

# ------------------------------------------------------------
# 3️⃣ Percentage-Based Accuracy Bands (Calories Only)
# ------------------------------------------------------------
print("\n📊 Percentage-Based Accuracy Bands (Calories)")
print("-" * 60)

# Calculate percentage error for calories (exclude zero actuals)
df["cal_pct_error"] = df.apply(
    lambda row: (abs(row["cal_error"]) / row["actual_cal"] * 100) if row["actual_cal"] != 0 else np.nan,
    axis=1
)

# Count within bands (excluding NaN from zero actuals)
valid_pct_errors = df["cal_pct_error"].dropna()
total_valid = len(valid_pct_errors)

within_10pct = (valid_pct_errors <= 10).sum() / total_valid * 100
within_20pct = (valid_pct_errors <= 20).sum() / total_valid * 100
within_30pct = (valid_pct_errors <= 30).sum() / total_valid * 100

print(f"Within ±10%:  {within_10pct:.1f}% of dishes")
print(f"Within ±20%:  {within_20pct:.1f}% of dishes")
print(f"Within ±30%:  {within_30pct:.1f}% of dishes")
print(f"\n(Based on {total_valid} dishes with non-zero actual calories)")

# ------------------------------------------------------------
# 4️⃣ Simple vs Mixed Dish Analysis
# ------------------------------------------------------------
print("\n📊 Simple vs Mixed Dish Analysis (Calories)")
print("-" * 60)

# Count ingredients using semicolons
df["ingredient_count"] = df["actual_ingredients"].apply(
    lambda x: x.count(";") + 1 if x else 0
)

# Classify dishes
df["dish_complexity"] = df["ingredient_count"].apply(
    lambda x: "Simple (1-3 ingredients)" if x <= 3 else "Mixed (4+ ingredients)"
)

# Group analysis
simple_dishes = df[df["dish_complexity"] == "Simple (1-3 ingredients)"]
mixed_dishes = df[df["dish_complexity"] == "Mixed (4+ ingredients)"]

simple_count = len(simple_dishes)
mixed_count = len(mixed_dishes)
simple_mae = simple_dishes["cal_error"].abs().mean()
mixed_mae = mixed_dishes["cal_error"].abs().mean()

print(f"\nSimple dishes (1-3 ingredients):")
print(f"  Count:  {simple_count}")
print(f"  MAE:    {simple_mae:.2f} kcal")

print(f"\nMixed dishes (4+ ingredients):")
print(f"  Count:  {mixed_count}")
print(f"  MAE:    {mixed_mae:.2f} kcal")

# T-test comparing calorie errors between groups
from scipy.stats import ttest_ind

t_stat, p_value = ttest_ind(
    simple_dishes["cal_error"].abs(),
    mixed_dishes["cal_error"].abs()
)

print(f"\nIndependent t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value:     {p_value:.4f}")

if p_value < 0.05:
    print(f"  Result: Statistically significant difference (p < 0.05)")
else:
    print(f"  Result: No statistically significant difference (p ≥ 0.05)")


print("\n📋 DATA PREPARATION SUMMARY")
print("-" * 60)

total_dishes = len(df)

print(f"\nTotal dishes analysed: {total_dishes}")
print("\nData cleaning steps applied:")
print("  • Duplicate records removed")
print("  • Rows with missing nutrient values (NaN) removed")
print("  • All numeric columns validated and converted")
print("\nEvaluation approach:")
print("  • Analysis conducted at plate (dish) level")
print("  • Ground truth: Lab-validated nutritional measurements")
print("  • Predictions: Computer vision model estimates")
print("  • Error metric: Prediction minus actual values")

print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60 + "\n")