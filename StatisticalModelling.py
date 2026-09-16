"""
STEP 2 — Load and Prepare the Data
======================================================================
Loads the Phase2_ML_Ready_Data sheet, drops identifiers/PII, and builds
the X (features) and y (target) arrays used in every later step.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Point this at your actual filename (same folder as this script)
SRC = "Loan_Eligibility_Phase3_Full__1_EDA_ANALYSIS_FINAL_UPDATION.xlsx"
SHEET = "Phase2_ML_Ready_Data"
TARGET = "Loan_Status_Encoded"

#2. Load the sheet 
df = pd.read_excel(SRC, sheet_name=SHEET)
print(f"Loaded {SHEET}: {df.shape[0]} rows x {df.shape[1]} columns")
print(df.head())

#3. Drop Applicant_ID and any other non-numeric identifier columns
drop_cols = ["Applicant_ID"]

# Convert any remaining Yes/No text columns to 1/0
obj_cols = [c for c in df.columns if df[c].dtype == "object" and c not in drop_cols]
for c in obj_cols:
    uniq = set(df[c].dropna().unique())
    if uniq <= {"Yes", "No"}:
        df[c] = df[c].map({"Yes": 1, "No": 0})

# Drop anything still non-numeric (safety net)
remaining_obj = [c for c in df.columns if df[c].dtype == "object" and c not in drop_cols]
drop_cols += remaining_obj

# 4. Build X (features) and y (target)
X = df.drop(columns=drop_cols + [TARGET])
y = df[TARGET].astype(int)

print(f"\nFeature matrix X: {X.shape[0]} rows x {X.shape[1]} columns")
print(f"Target y: {y.shape[0]} rows")

#5. Confirm there are no missing values left

missing = X.isnull().sum().sum()
print(f"\nMissing values in X: {missing}")
if missing == 0:
    print("Clean — no missing values. Ready for Step 3 (train/validation/test split).")
else:
    print("WARNING: missing values found — check your source file before continuing.")

print("\nTarget class distribution:")
print(y.value_counts().sort_index())

# ---- 6. Same distribution, shown with readable class names ----
# (y itself stays numeric — models need numbers — this is just for display)
CLASS_LABELS = {0: "Rejected", 1: "Pending", 2: "Manual Review", 3: "Approved"}
readable_counts = y.value_counts().sort_index().rename(index=CLASS_LABELS)
print("\nTarget class distribution (labelled):")
print(readable_counts)

RANDOM_STATE = 42  # fixed seed so your split is reproducible every time you run this

#The first split — carving off 70% for training
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

#The second split — dividing the leftover 30% in half
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print(f"Train:      {X_train.shape[0]:>4} rows ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation: {X_val.shape[0]:>4} rows ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test:       {X_test.shape[0]:>4} rows ({X_test.shape[0]/len(X)*100:.1f}%)")
print("\n ========================================================");
print("\n FULL /TRAIN / VALIDATION / TEST SPLIT ");
print("\n ========================================================");
check = pd.DataFrame({
    "Full_%": (y.value_counts(normalize=True) * 100).round(1),
    "Train_%": (y_train.value_counts(normalize=True) * 100).round(1),
    "Val_%": (y_val.value_counts(normalize=True) * 100).round(1),
    "Test_%": (y_test.value_counts(normalize=True) * 100).round(1),
    }).rename(index=CLASS_LABELS).sort_index()
print(check)

print("\n Step 3 complete. X_train, X_val, X_test, y_train, y_val, y_test are ready for Step 4 (scaling).");
print("\n ========================================================");
