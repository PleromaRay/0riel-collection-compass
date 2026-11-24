"""Clean the raw customers export (Airbyte load) for Collections Compass.

Usage (from the project folder):
    python clean_customers.py data/raw/customers_raw.csv data/processed/customers_clean.csv
"""
import sys
from pathlib import Path

import pandas as pd

RAW_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/customers_raw.csv")
OUT_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/processed/customers_clean.csv")
REFERENCE_DATE = pd.Timestamp("2026-11-22")  # date of the latest loan-balance snapshot

df = pd.read_csv(RAW_PATH)
print(f"Loaded {len(df):,} rows and {df.shape[1]} columns")

# 1. Drop the loading-tool metadata columns (they all start with _AIRBYTE or _AB)
meta_cols = [c for c in df.columns if c.startswith(("_AIRBYTE", "_AB"))]
df = df.drop(columns=meta_cols)

# 2. Tidy column names to lower case
df.columns = df.columns.str.lower()

# 3. Strip stray spaces from text columns
text_cols = df.select_dtypes(exclude="number").columns
df[text_cols] = df[text_cols].apply(lambda s: s.str.strip())

# 4. Signup date: most rows are YYYY-MM-DD, some are DD/MM/YYYY, so parse each format
iso = pd.to_datetime(df["signup_date"], format="%Y-%m-%d", errors="coerce")
uk = pd.to_datetime(df["signup_date"], format="%d/%m/%Y", errors="coerce")
df["signup_date"] = iso.fillna(uk)
assert df["signup_date"].notna().all(), "Some signup dates could not be parsed"

# 5. Y/N flags to 1/0
for col in ["sms_opt_in", "phone_valid"]:
    df[col] = df[col].map({"Y": 1, "N": 0}).astype("int8")

# 6. Missing categories become an explicit 'Unknown' level (missingness can carry signal)
df["employment_status"] = df["employment_status"].fillna("Unknown")
df["monthly_income_band"] = df["monthly_income_band"].fillna("Unknown")

# 7. Payday: about 55% missing, so keep it as nullable and add a flag
df["payday_known"] = df["payday_day_of_month"].notna().astype("int8")
df["payday_day_of_month"] = df["payday_day_of_month"].astype("Int64")

# 8. Simple derived fields
df["age_at_reference"] = REFERENCE_DATE.year - df["year_of_birth"]
df["tenure_days"] = (REFERENCE_DATE - df["signup_date"]).dt.days
income_order = ["Under £1,000", "£1,000 - £1,999", "£2,000 - £2,999", "£3,000 and over", "Unknown"]
df["monthly_income_band"] = pd.Categorical(df["monthly_income_band"], categories=income_order, ordered=False)

# 9. Checks
assert df["customer_id"].is_unique, "Duplicate customer IDs found"
assert (df["age_at_reference"] >= 18).all(), "Customers under 18 found"
assert (df["tenure_days"] >= 0).all(), "Signup dates after the reference date found"

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"Saved {len(df):,} rows and {df.shape[1]} columns to {OUT_PATH}")
print(df.isna().sum()[lambda s: s > 0])
