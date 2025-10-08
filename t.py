import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("ufc_data/training_dataset_cache.csv")

# --- Step 1: Convert event_date to datetime and filter ---
df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

# Keep only rows where year > 2010
before_rows = len(df)
df = df[df["event_date"].dt.year > 2010]
after_rows = len(df)

print(f"Rows before filter: {before_rows}")
print(f"Rows after filter (event_date > 2010): {after_rows}")

# --- Step 2: Summary of missing values ---
missing_counts = df.isnull().sum()
missing_percentage = (missing_counts / len(df)) * 100

# Combine into a summary table
missing_summary = pd.DataFrame({
    "Missing Values": missing_counts,
    "Percentage": missing_percentage.round(2)
}).sort_values(by="Missing Values", ascending=False)

print("\nMissing Data Summary (Filtered: event_date > 2010):")
print(missing_summary)

# --- Step 3: Total complete rows ---
complete_rows = df.dropna().shape[0]
print(f"\nTotal rows without any missing values: {complete_rows} out of {len(df)}")

# --- Step 4: Plot side-by-side charts ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left chart: counts
missing_summary[missing_summary["Missing Values"] > 0]["Missing Values"].plot(
    kind="bar", color="steelblue", ax=axes[0]
)
axes[0].set_title("Missing Values by Column (Count)")
axes[0].set_ylabel("Number of Missing Entries")
axes[0].set_xlabel("Columns")
axes[0].tick_params(axis="x", rotation=45)

# Right chart: percentages
missing_summary[missing_summary["Percentage"] > 0]["Percentage"].plot(
    kind="bar", color="tomato", ax=axes[1]
)
axes[1].set_title("Missing Values by Column (%)")
axes[1].set_ylabel("Percentage of Missing Entries")
axes[1].set_xlabel("Columns")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
