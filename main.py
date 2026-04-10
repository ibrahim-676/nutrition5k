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
#  SIMPLE VISUALISATION BLOCK — Nutrition5K Dataset
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
# ------------------------------------------------------------
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




# ------------------------------------------------------------
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