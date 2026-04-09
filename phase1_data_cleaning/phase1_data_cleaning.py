# =============================================================
# RETAIL & LOGISTICS PYTHON ANALYTICS
# Phase 1 — Data Cleaning
# Dataset: Superstore Sales (Kaggle)
# Stakeholder: Analytics Lead
# =============================================================

import pandas as pd
import numpy as np

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------

df = pd.read_csv('superstore.csv', encoding='latin-1')

print("=" * 50)
print("Raw Data Preview")
print("=" * 50)
print(df.head())
print(f"\nInitial shape: {df.shape}")


# -------------------------------------------------------------
# QUESTION 1
# Stakeholder: "Can you check the dataset for missing values
# and give me a summary of where the gaps are — which columns
# are affected, how many values are missing, and whether it's
# significant enough to impact our analysis?"
# -------------------------------------------------------------

profile = pd.DataFrame({
    'dtype'        : df.dtypes,
    'null_count'   : df.isnull().sum(),
    'null_pct'     : round(df.isnull().sum() * 100.0 / len(df), 2),
    'unique_values': df.nunique()
}).sort_values('null_pct', ascending=False)

print("\n" + "=" * 50)
print("Q1 — Missing Value Audit")
print("=" * 50)
print(profile)

# Finding: Zero missing values across all 21 columns.
# No imputation needed.


# -------------------------------------------------------------
# QUESTION 2
# Stakeholder: "Ship Date and Order Date are showing as object
# dtype. Can you fix the date columns, create a shipping_days
# column, and give me average shipping time by Ship Mode?
# Is Standard Class as slow as the operations team claims?"
# -------------------------------------------------------------

# Fix dtypes
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date']  = pd.to_datetime(df['Ship Date'])

# Create shipping_days feature
df['shipping_days'] = (df['Ship Date'] - df['Order Date']).dt.days

# Average shipping days by Ship Mode
avg_shipping = round(
    df.groupby('Ship Mode')['shipping_days'].mean(), 2
).reset_index()
avg_shipping.columns = ['Ship Mode', 'Avg Shipping Days']

print("\n" + "=" * 50)
print("Q2 — Average Shipping Days by Ship Mode")
print("=" * 50)
print(avg_shipping.to_string(index=False))

# Finding:
# First Class    — 2.18 days
# Same Day       — 0.04 days
# Second Class   — 3.24 days
# Standard Class — 5.01 days
# Insight: Standard Class concern from operations is validated.


# -------------------------------------------------------------
# QUESTION 3
# Stakeholder: "I've heard we've had duplicate orders creep
# into our system. Can you check for duplicates, remove them,
# and confirm the shape before and after?"
# -------------------------------------------------------------

print("\n" + "=" * 50)
print("Q3 — Duplicate Row Check")
print("=" * 50)

print(f"Shape before dedup : {df.shape}")
print(f"Duplicate rows found: {df.duplicated().sum()}")

df = df.drop_duplicates().reset_index(drop=True)

print(f"Shape after dedup  : {df.shape}")

# Finding: 0 duplicates found. Shape remains 9,994 rows.


# -------------------------------------------------------------
# QUESTION 4
# Stakeholder: "Can you standardise all column names and give
# me a final dataframe summary — dtypes and shape — so I can
# sign off on the data before we hand it to the analytics team?"
# -------------------------------------------------------------

# Standardise column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
)

print("\n" + "=" * 50)
print("Q4 — Final Data Summary After Cleaning")
print("=" * 50)
print(f"Shape  : {df.shape}")
print(f"\nDtypes :\n{df.dtypes}")

# Finding:
# 22 columns, all snake_case
# Datetimes correct, numerics correct
# shipping_days added as int64


# -------------------------------------------------------------
# SAVE CLEANED DATA FOR NEXT PHASES
# -------------------------------------------------------------

df.to_csv('superstore_cleaned.csv', index=False)
print("\nCleaned dataset saved as superstore_cleaned.csv")