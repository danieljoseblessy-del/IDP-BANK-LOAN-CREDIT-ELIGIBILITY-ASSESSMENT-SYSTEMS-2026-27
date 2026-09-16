

"""
STEP 2 — Load and Prepare the Data
======================================================================
Loads the Phase2_ML_Ready_Data sheet, drops identifiers/PII, and builds
the X (features) and y (target) arrays used in every later step.
"""
import shap
import scipy.stats
import pandas as pd
import numpy as np

from scipy.stats import (
    chi2_contingency,
    f_oneway,
    ttest_ind,
    pearsonr,
    spearmanr
)

import statsmodels.api as sm

# ---- 1. Point this at your actual filename (same folder as this script) ----
SRC = "Loan_Eligibility_Phase3_Full__1_EDA_ANALYSIS_FINAL_UPDATION.xlsx"
SHEET = "Phase2_ML_Ready_Data"
TARGET = "Loan_Status_Encoded"

# ---- 2. Load the sheet ----
df = pd.read_excel(SRC, sheet_name=SHEET)
print(f"Loaded {SHEET}: {df.shape[0]} rows x {df.shape[1]} columns")
print(df.head())





#PART 6 - Statistical Modelling  
# ======================================================================
# PART A — FORMAL HYPOTHESIS TESTING
# WELCH INDEPENDENT-SAMPLES T-TEST
# ======================================================================

print("\n" + "=" * 78)
print("FORMAL HYPOTHESIS TESTING — WELCH T-TEST")
print("=" * 78)

# ----------------------------------------------------------------------
# Objective:
# Compare the mean Credit_Score of:
#   Group 1 = Approved applicants
#   Group 2 = Not Approved applicants
#
# H0:
# Mean Credit Score of Approved = Mean Credit Score of Not Approved
#
# H1:
# Mean Credit Score of Approved != Mean Credit Score of Not Approved
#
# Significance level:
# alpha = 0.05
# ----------------------------------------------------------------------

ALPHA = 0.05
APPROVED_CODE = 3

# Check whether Credit_Score exists
if "Credit_Score" not in df.columns:

    print("\nERROR: 'Credit_Score' column was not found.")
    print("T-test cannot be performed.")

else:

    # Approved applicants
    approved_scores = pd.to_numeric(
        df.loc[
            df[TARGET] == APPROVED_CODE,
            "Credit_Score"
        ],
        errors="coerce"
    ).dropna()

    # Not approved = Rejected, Pending, Manual Review
    not_approved_scores = pd.to_numeric(
        df.loc[
            df[TARGET] != APPROVED_CODE,
            "Credit_Score"
        ],
        errors="coerce"
    ).dropna()

    print("\nGroup sizes:")
    print("Approved applicants     :", len(approved_scores))
    print("Not approved applicants :", len(not_approved_scores))

    if len(approved_scores) >= 2 and len(not_approved_scores) >= 2:

        # Welch's t-test
        t_statistic, p_value = ttest_ind(
            approved_scores,
            not_approved_scores,
            equal_var=False,
            nan_policy="omit"
        )

        print("\nMean Credit Score:")
        print(
            "Approved     :",
            round(approved_scores.mean(), 4)
        )
        print(
            "Not Approved :",
            round(not_approved_scores.mean(), 4)
        )

        print("\nT-Test Result:")
        print(
            "T-Statistic :",
            round(t_statistic, 4)
        )
        print(
            "P-Value     :",
            f"{p_value:.6f}"
        )
        print(
            "Alpha       :",
            ALPHA
        )

        # Decision
        if p_value < ALPHA:

            print("\nDecision: Reject H0.")

            print(
                "Conclusion: There is a statistically significant "
                "difference in mean Credit Score between approved "
                "and not-approved applicants."
            )

        else:

            print("\nDecision: Fail to reject H0.")

            print(
                "Conclusion: There is no statistically significant "
                "difference in mean Credit Score between approved "
                "and not-approved applicants."
            )

        # Save result
        hypothesis_test_result = pd.DataFrame({
            "Test": ["Welch Independent-Samples T-Test"],
            "Variable": ["Credit_Score"],
            "Group_1": ["Approved"],
            "Group_2": ["Not Approved"],
            "Group_1_Mean": [approved_scores.mean()],
            "Group_2_Mean": [not_approved_scores.mean()],
            "Group_1_N": [len(approved_scores)],
            "Group_2_N": [len(not_approved_scores)],
            "T_Statistic": [t_statistic],
            "P_Value": [p_value],
            "Significance_Level": [ALPHA],
            "Decision": [
                "Reject H0"
                if p_value < ALPHA
                else "Fail to reject H0"
            ]
        })

        hypothesis_test_result.to_excel(
            "statistical_hypothesis_ttest_results.xlsx",
            index=False
        )

        print(
            "\nSaved: statistical_hypothesis_ttest_results.xlsx"
        )

    else:

        print(
            "\nT-test could not be performed because one "
            "or both groups contain fewer than 2 observations."
        )




# ======================================================================
# STATISTICAL ANALYSIS
# CHI-SQUARE TEST AND ONE-WAY ANOVA
# ======================================================================
#
# This section is performed after loading and preparing the dataset
# and BEFORE the ML train/validation/test split.
#
# The target variable is:
#     Loan_Status_Encoded
#
# Chi-Square:
#     Categorical feature vs Loan_Status_Encoded
#
# ANOVA:
#     Numerical feature vs Loan_Status_Encoded
# ======================================================================

from scipy.stats import chi2_contingency, f_oneway


print("\n")
print("=" * 78)
print("STATISTICAL ANALYSIS")
print("=" * 78)


# ======================================================================
# PART B — CHI-SQUARE TEST OF INDEPENDENCE
# ======================================================================
#
# Purpose:
# Determine whether a categorical feature has a statistically
# significant association with Loan_Status_Encoded.
#
# H0 (Null Hypothesis):
# The categorical feature and loan status are independent.
#
# H1 (Alternative Hypothesis):
# The categorical feature and loan status are associated.
#
# Significance level:
# alpha = 0.05
# ======================================================================

print("\n" + "=" * 78)
print("CHI-SQUARE TEST OF INDEPENDENCE")
print("=" * 78)

ALPHA = 0.05

chi_square_results = []


# ----------------------------------------------------------------------
# Select categorical columns
# ----------------------------------------------------------------------

categorical_columns = df.select_dtypes(
    include=["object", "category", "bool", "string"]
).columns.tolist()

categorical_columns = [
    col for col in categorical_columns
    if col not in ["Applicant_ID", TARGET]
]

# Remove the target if it happens to be categorical
categorical_columns = [
    col for col in categorical_columns
    if col != TARGET
]


print("\nCategorical features available for Chi-Square testing:")

for col in categorical_columns:
    print(" -", col)


# ----------------------------------------------------------------------
# Perform Chi-Square test for every categorical feature
# ----------------------------------------------------------------------

for column in categorical_columns:

    try:

        # Create contingency table
        contingency_table = pd.crosstab(
            df[column],
            df[TARGET]
        )

        # Skip variables that do not have enough categories
        if contingency_table.shape[0] < 2:
            continue

        if contingency_table.shape[1] < 2:
            continue


        # Perform Chi-Square test
        chi2_statistic, p_value, degrees_of_freedom, expected = (
            chi2_contingency(contingency_table)
        )


        # Store result
        chi_square_results.append({
            "Feature": column,
            "Chi_Square_Statistic": chi2_statistic,
            "Degrees_of_Freedom": degrees_of_freedom,
            "P_Value": p_value,
            "Significance_Level": ALPHA,
            "Significant": (
                "Yes" if p_value < ALPHA else "No"
            )
        })


        # Display result
        print("\nFeature:", column)
        print(
            "Chi-Square Statistic:",
            round(chi2_statistic, 4)
        )

        print(
            "Degrees of Freedom:",
            degrees_of_freedom
        )

        print(
            "p-value:",
            f"{p_value:.6f}"
        )


        if p_value < ALPHA:

            print(
                "Conclusion: Reject H0 — "
                "significant association exists."
            )

        else:

            print(
                "Conclusion: Fail to reject H0 — "
                "no statistically significant association detected."
            )


    except Exception as e:

        print(
            f"\nChi-Square test failed for {column}: {e}"
        )


# ----------------------------------------------------------------------
# Save Chi-Square results
# ----------------------------------------------------------------------

chi_square_results_df = pd.DataFrame(
    chi_square_results
)


if not chi_square_results_df.empty:

    chi_square_results_df = chi_square_results_df.sort_values(
        by="P_Value",
        ascending=True
    ).reset_index(drop=True)


    chi_square_results_df.to_excel(
        "statistical_chi_square_results.xlsx",
        index=False
    )


print("\n" + "-" * 78)
print("CHI-SQUARE SUMMARY")
print("-" * 78)

if not chi_square_results_df.empty:

    print(
        chi_square_results_df.to_string(index=False)
    )

    print(
        "\nSaved: statistical_chi_square_results.xlsx"
    )

else:

    print(
        "No valid categorical variables were available "
        "for Chi-Square testing."
    )


# ======================================================================
# PART C — ONE-WAY ANOVA
# ======================================================================
#
# Purpose:
# Determine whether the mean of a numerical feature differs
# significantly across the loan-status groups.
#
# H0:
# All group means are equal.
#
# H1:
# At least one group mean is different.
#
# Significance level:
# alpha = 0.05
# ======================================================================

print("\n" + "=" * 78)
print("ONE-WAY ANOVA")
print("=" * 78)


anova_results = []


# ----------------------------------------------------------------------
# Select numerical columns
# ----------------------------------------------------------------------

numerical_columns = df.select_dtypes(
    include=[np.number]
).columns.tolist()


# Remove the target variable
numerical_columns = [
    col for col in numerical_columns
    if col != TARGET
]


print("\nNumerical features available for ANOVA:")

for col in numerical_columns:
    print(" -", col)


# ----------------------------------------------------------------------
# Identify loan-status groups
# ----------------------------------------------------------------------

loan_status_groups = sorted(
    df[TARGET].dropna().unique()
)


print("\nLoan status groups:")
print(loan_status_groups)


# ----------------------------------------------------------------------
# Perform One-Way ANOVA for every numerical feature
# ----------------------------------------------------------------------

for column in numerical_columns:

    try:

        groups = []

        valid_group_labels = []


        for status in loan_status_groups:

            group_data = df.loc[
                df[TARGET] == status,
                column
            ].dropna()


            # A group needs at least two observations
            if len(group_data) >= 2:

                groups.append(
                    group_data
                )

                valid_group_labels.append(
                    status
                )


        # Need at least two groups
        if len(groups) < 2:

            continue


        # Perform one-way ANOVA
        f_statistic, p_value = f_oneway(
            *groups
        )


        # Store result
        anova_results.append({
            "Feature": column,
            "F_Statistic": f_statistic,
            "P_Value": p_value,
            "Significance_Level": ALPHA,
            "Significant": (
                "Yes" if p_value < ALPHA else "No"
            )
        })


        # Display result
        print("\nFeature:", column)

        print(
            "F-statistic:",
            round(f_statistic, 4)
        )

        print(
            "p-value:",
            f"{p_value:.6f}"
        )


        if p_value < ALPHA:

            print(
                "Conclusion: Reject H0 — "
                "a significant difference exists "
                "between at least two group means."
            )

        else:

            print(
                "Conclusion: Fail to reject H0 — "
                "no statistically significant difference "
                "between group means was detected."
            )


    except Exception as e:

        print(
            f"\nANOVA failed for {column}: {e}"
        )


# ----------------------------------------------------------------------
# Save ANOVA results
# ----------------------------------------------------------------------

anova_results_df = pd.DataFrame(
    anova_results
)


if not anova_results_df.empty:

    anova_results_df = anova_results_df.sort_values(
        by="P_Value",
        ascending=True
    ).reset_index(drop=True)


    anova_results_df.to_excel(
        "statistical_anova_results.xlsx",
        index=False
    )


print("\n" + "-" * 78)
print("ANOVA SUMMARY")
print("-" * 78)

if not anova_results_df.empty:

    print(
        anova_results_df.to_string(index=False)
    )

    print(
        "\nSaved: statistical_anova_results.xlsx"
    )

else:

    print(
        "No valid numerical variables were available "
        "for ANOVA."
    )


# ======================================================================
# STATISTICAL ANALYSIS COMPLETE
# ======================================================================

print("\n" + "=" * 78)
print("STATISTICAL ANALYSIS COMPLETE")
print("=" * 78)

print(
    "\nChi-Square results saved to:"
)

print(
    "statistical_chi_square_results.xlsx"
)

print(
    "\nANOVA results saved to:"
)

print(
    "statistical_anova_results.xlsx"
)

print("\nProceeding to CORRELATION ANALYSIS : PEARSON AND SPEARMAN CORRELATION...")


# ======================================================================
# PART D — CORRELATION ANALYSIS
# PEARSON AND SPEARMAN CORRELATION
# ======================================================================

print("\n" + "=" * 78)
print("CORRELATION ANALYSIS")
print("=" * 78)

# ----------------------------------------------------------------------
# Select numerical variables
# ----------------------------------------------------------------------

correlation_columns = df.select_dtypes(
    include=[np.number]
).columns.tolist()

# Remove target and identifier
correlation_columns = [
    col for col in correlation_columns
    if col not in ["Applicant_ID", TARGET]
]

print("\nNumerical variables used for correlation:")

for col in correlation_columns:
    print(" -", col)


if len(correlation_columns) >= 2:

    # --------------------------------------------------------------
    # Pearson correlation matrix
    # --------------------------------------------------------------

    pearson_matrix = df[
        correlation_columns
    ].corr(method="pearson")

    print("\n" + "-" * 78)
    print("PEARSON CORRELATION MATRIX")
    print("-" * 78)

    print(
        pearson_matrix.round(4).to_string()
    )

    # Save Pearson matrix
    pearson_matrix.to_excel(
        "statistical_pearson_correlation_matrix.xlsx"
    )

    # --------------------------------------------------------------
    # Spearman correlation matrix
    # --------------------------------------------------------------

    spearman_matrix = df[
        correlation_columns
    ].corr(method="spearman")

    print("\n" + "-" * 78)
    print("SPEARMAN CORRELATION MATRIX")
    print("-" * 78)

    print(
        spearman_matrix.round(4).to_string()
    )

    # Save Spearman matrix
    spearman_matrix.to_excel(
        "statistical_spearman_correlation_matrix.xlsx"
    )

    # --------------------------------------------------------------
    # Pairwise correlation significance results
    # --------------------------------------------------------------

    correlation_results = []

    for i in range(len(correlation_columns)):

        for j in range(i + 1, len(correlation_columns)):

            variable_1 = correlation_columns[i]
            variable_2 = correlation_columns[j]

            pair = df[
                [variable_1, variable_2]
            ].dropna()

            if len(pair) < 3:
                continue

            # Pearson
            pearson_r, pearson_p = pearsonr(
                pair[variable_1],
                pair[variable_2]
            )

            # Spearman
            spearman_rho, spearman_p = spearmanr(
                pair[variable_1],
                pair[variable_2]
            )

            correlation_results.append({
                "Variable_1": variable_1,
                "Variable_2": variable_2,
                "N": len(pair),
                "Pearson_r": pearson_r,
                "Pearson_P_Value": pearson_p,
                "Spearman_Rho": spearman_rho,
                "Spearman_P_Value": spearman_p,
                "Significance_Level": ALPHA
            })

    correlation_results_df = pd.DataFrame(
        correlation_results
    )

    if not correlation_results_df.empty:

        correlation_results_df = correlation_results_df.sort_values(
            by="Pearson_P_Value",
            ascending=True
        ).reset_index(drop=True)

        correlation_results_df.to_excel(
            "statistical_correlation_analysis_results.xlsx",
            index=False
        )

        print("\nPairwise correlation significance results:")
        print(
            correlation_results_df.to_string(index=False)
        )

        print(
            "\nSaved: "
            "statistical_correlation_analysis_results.xlsx"
        )

    else:

        print(
            "\nNo valid pairwise correlations were available."
        )

else:

    print(
        "\nCorrelation analysis requires at least "
        "two numerical variables."
    )


# ======================================================================
# PART E — MULTIPLE LINEAR REGRESSION
# ======================================================================

print("\n" + "=" * 78)
print("MULTIPLE LINEAR REGRESSION")
print("=" * 78)

# ----------------------------------------------------------------------
# Objective:
# Explain/predict Credit_Score using the other numerical variables.
#
# Dependent variable:
#     Credit_Score
#
# Independent variables:
#     Other numerical variables excluding:
#       - Applicant_ID
#       - Loan_Status_Encoded
#       - Credit_Score
#
# Statistical model:
#
# Credit_Score = beta0 + beta1X1 + beta2X2 + ... + error
# ----------------------------------------------------------------------

REGRESSION_TARGET = "Credit_Score"

if REGRESSION_TARGET not in df.columns:

    print(
        "\nERROR: 'Credit_Score' column was not found."
    )

else:

    # --------------------------------------------------------------
    # Select numerical predictor variables
    # --------------------------------------------------------------

    regression_predictors = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    regression_predictors = [
        col for col in regression_predictors
        if col not in [
            "Applicant_ID",
            TARGET,
            REGRESSION_TARGET
        ]
    ]

    print("\nRegression dependent variable:")
    print("-", REGRESSION_TARGET)

    print("\nRegression predictor variables:")

    for col in regression_predictors:
        print("-", col)

    if len(regression_predictors) == 0:

        print(
            "\nMultiple regression cannot be performed because "
            "no numerical predictor variables are available."
        )

    else:

        # ----------------------------------------------------------
        # Build regression dataset
        # ----------------------------------------------------------

        regression_data = df[
            [REGRESSION_TARGET] + regression_predictors
        ].copy()

        # Convert everything to numeric
        for col in regression_data.columns:

            regression_data[col] = pd.to_numeric(
                regression_data[col],
                errors="coerce"
            )

        # Remove rows with missing values
        regression_data = regression_data.dropna()

        print(
            f"\nRegression observations used: "
            f"{len(regression_data)}"
        )

        # ----------------------------------------------------------
        # Dependent variable
        # ----------------------------------------------------------

        regression_y = regression_data[
            REGRESSION_TARGET
        ]

        # ----------------------------------------------------------
        # Independent variables
        # ----------------------------------------------------------

        regression_X = regression_data[
            regression_predictors
        ]

        # Add intercept
        regression_X = sm.add_constant(
            regression_X,
            has_constant="add"
        )

        # ----------------------------------------------------------
        # Fit Ordinary Least Squares regression
        # ----------------------------------------------------------

        multiple_regression_model = sm.OLS(
            regression_y,
            regression_X
        ).fit()

        # ----------------------------------------------------------
        # Display complete statistical summary
        # ----------------------------------------------------------

        print(
            "\nMULTIPLE REGRESSION SUMMARY"
        )

        print(
            multiple_regression_model.summary()
        )

        # ----------------------------------------------------------
        # Extract model statistics
        # ----------------------------------------------------------

        print("\n" + "-" * 78)
        print("MODEL STATISTICS")
        print("-" * 78)

        print(
            "R-Squared:",
            round(
                multiple_regression_model.rsquared,
                4
            )
        )

        print(
            "Adjusted R-Squared:",
            round(
                multiple_regression_model.rsquared_adj,
                4
            )
        )

        print(
            "F-Statistic:",
            round(
                multiple_regression_model.fvalue,
                4
            )
        )

        print(
            "F-Test P-Value:",
            f"{multiple_regression_model.f_pvalue:.6f}"
        )

        # ----------------------------------------------------------
        # Regression coefficient table
        # ----------------------------------------------------------

        regression_coefficients = pd.DataFrame({
            "Variable":
                multiple_regression_model.params.index,

            "Coefficient":
                multiple_regression_model.params.values,

            "Std_Error":
                multiple_regression_model.bse.values,

            "T_Statistic":
                multiple_regression_model.tvalues.values,

            "P_Value":
                multiple_regression_model.pvalues.values,

            "CI_Lower_95":
                multiple_regression_model.conf_int()[0].values,

            "CI_Upper_95":
                multiple_regression_model.conf_int()[1].values
        })

        # Add significance decision
        regression_coefficients[
            "Significant_at_0.05"
        ] = np.where(
            regression_coefficients["P_Value"] < ALPHA,
            "Yes",
            "No"
        )

        print("\nRegression coefficients:")
        print(
            regression_coefficients.to_string(
                index=False
            )
        )

        # ----------------------------------------------------------
        # Overall model conclusion
        # ----------------------------------------------------------

        if multiple_regression_model.f_pvalue < ALPHA:

            print(
                "\nOverall Model Conclusion:"
            )

            print(
                "Reject H0 — the regression model is "
                "statistically significant at the 5% level."
            )

        else:

            print(
                "\nOverall Model Conclusion:"
            )

            print(
                "Fail to reject H0 — the regression model "
                "is not statistically significant at the 5% level."
            )

        # ----------------------------------------------------------
        # Save regression results
        # ----------------------------------------------------------

        regression_coefficients.to_excel(
            "statistical_multiple_regression_coefficients.xlsx",
            index=False
        )

        regression_model_statistics = pd.DataFrame({
            "Statistic": [
                "R-Squared",
                "Adjusted R-Squared",
                "F-Statistic",
                "F-Test P-Value",
                "Significance Level"
            ],

            "Value": [
                multiple_regression_model.rsquared,
                multiple_regression_model.rsquared_adj,
                multiple_regression_model.fvalue,
                multiple_regression_model.f_pvalue,
                ALPHA
            ]
        })

        regression_model_statistics.to_excel(
            "statistical_multiple_regression_model_statistics.xlsx",
            index=False
        )

        # Save textual regression summary
        with open(
            "statistical_multiple_regression_summary.txt",
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                multiple_regression_model.summary().as_text()
            )

        print(
            "\nSaved:"
        )

        print(
            "- statistical_multiple_regression_coefficients.xlsx"
        )

        print(
            "- statistical_multiple_regression_model_statistics.xlsx"
        )

        print(
            "- statistical_multiple_regression_summary.txt"
        )

# ---- 3. Drop Applicant_ID and any other non-numeric identifier columns ----
drop_cols = ["Applicant_ID"]

# Convert any remaining Yes/No text columns to 1/0.
# NOTE: pandas 3.x stores text columns as dtype "str" instead of the older
# "object" dtype, so we check for both to work on any pandas version.
def is_text_column(series):
    return series.dtype == "object" or str(series.dtype) in ("string", "str")

obj_cols = [c for c in df.columns if is_text_column(df[c]) and c not in drop_cols]
for c in obj_cols:
    uniq = set(df[c].dropna().unique())
    if uniq <= {"Yes", "No"}:
        df[c] = df[c].map({"Yes": 1, "No": 0})

# Drop anything still non-numeric (safety net)
remaining_obj = [c for c in df.columns if is_text_column(df[c]) and c not in drop_cols]
drop_cols += remaining_obj


# ---- 4. Build X (features) and y (target) ----
X = df.drop(columns=drop_cols + [TARGET])
y = df[TARGET].astype(int)

print(f"\nFeature matrix X: {X.shape[0]} rows x {X.shape[1]} columns")
print(f"Target y: {y.shape[0]} rows")


# ---- 5. Confirm there are no missing values left ----
missing = X.isnull().sum().sum()
print(f"\nMissing values in X: {missing}")
if missing == 0:
    print("Clean — no missing values. Ready for Step 3 (train/validation/test split).")
else:
    print("WARNING: missing values found — check your source file before continuing.")

print("\nTarget class distribution:")
print(y.value_counts().sort_index())
        

# ======================================================================
# STEP 3 — Stratified Train / Validation / Test Split (70% / 15% / 15%)
# ======================================================================
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42  # fixed seed so your split is reproducible every time you run this

# First split: 70% train, 30% "temp" (which becomes validation + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y  # keeps each class's proportion identical across the split
)

# Second split: cut the 30% temp evenly into 15% validation and 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("\n" + "=" * 60)
print("STEP 3 — TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)
print(f"Train:      {X_train.shape[0]:>4} rows ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation: {X_val.shape[0]:>4} rows ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test:       {X_test.shape[0]:>4} rows ({X_test.shape[0]/len(X)*100:.1f}%)")

CLASS_LABELS = {
    0: "Rejected",
    1: "Pending",
    2: "Manual Review",
    3: "Approved"
}

# ---- Sanity check: confirm each class kept the same proportion everywhere ----
print("\nClass balance check (should look nearly identical across all 3 columns):")
import pandas as pd
check = pd.DataFrame({
    "Full_%": (y.value_counts(normalize=True) * 100).round(1),
    "Train_%": (y_train.value_counts(normalize=True) * 100).round(1),
    "Val_%": (y_val.value_counts(normalize=True) * 100).round(1),
    "Test_%": (y_test.value_counts(normalize=True) * 100).round(1),
}).rename(index=CLASS_LABELS).sort_index()
print(check)

print("\nStep 3 complete. X_train, X_val, X_test, y_train, y_val, y_test are ready for Step 4 (scaling).")

# ======================================================================
# STEP 4 — Scale the Features
# ======================================================================
from sklearn.preprocessing import StandardScaler

# Fit the scaler ONLY on training data — never on validation or test.
# "Fitting" means the scaler learns each column's mean and standard
# deviation from X_train. Applying (transforming) that same learned
# scale to validation/test avoids leaking information from data the
# model isn't supposed to have seen yet.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)   # learn + apply, on train
X_val_scaled = scaler.transform(X_val)           # apply only, on validation
X_test_scaled = scaler.transform(X_test)         # apply only, on test

print("\n" + "=" * 60)
print("STEP 4 — FEATURE SCALING")
print("=" * 60)
print(f"X_train_scaled: {X_train_scaled.shape}")
print(f"X_val_scaled:   {X_val_scaled.shape}")
print(f"X_test_scaled:  {X_test_scaled.shape}")

# ---- Sanity check: scaled training data should have mean ~0, std ~1 ----
import numpy as np
print(f"\nTraining data after scaling — mean of first 5 columns (should be ~0):")
print(np.round(X_train_scaled[:, :5].mean(axis=0), 4))
print(f"Training data after scaling — std of first 5 columns (should be ~1):")
print(np.round(X_train_scaled[:, :5].std(axis=0), 4))

print("\nStep 4 complete. X_train_scaled, X_val_scaled, X_test_scaled are ready for Step 5 (cross-validation + hyperparameter tuning).")


# ======================================================================
# STEP 5 — Cross-Validation + Hyperparameter Tuning (RBI-aligned)
# ======================================================================
# RBI / regulatory alignment built into this step, not just claimed:
#
#  - STRATIFIED folds (StratifiedKFold, not plain KFold): every fold keeps
#    the same class proportions as the full dataset, so the rare Pending
#    (4.3%) and Rejected (10.2%) classes are never accidentally starved
#    out of a fold. This matters for RBI's Digital Lending Directions,
#    which expect non-discriminatory, evidence-based credit decisioning
#    that is not blind to minority outcome classes.
#
#  - Tuning runs ONLY on X_train_scaled/y_train. The validation and test
#    sets are never touched here, so no information leaks into model
#    selection - this is what makes the eventual test-set number in
#    Step 7 trustworthy for an audit rather than optimistic.
#
#  - A fixed RANDOM_STATE is used everywhere, so every fold, every search,
#    and every model fit is exactly reproducible on any machine - a basic
#    requirement for model governance / audit trails in regulated lending.
#
#  - Every model's winning hyperparameters are printed and can be saved,
#    so the final model's configuration is fully documented rather than
#    a black box - supporting the explainability expectation carried
#    through the rest of this project.
# ======================================================================

import time
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("NOTE: xgboost not found — skipping XGBoost. Run 'pip install xgboost' if you want it included.")

# 10-fold stratified CV for the robust performance estimate (RBI-style
# validation discipline: mean + spread across folds, not one lucky split)
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
# 5-fold stratified CV used inside the hyperparameter search itself,
# to keep runtime reasonable while still being class-aware
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

model_specs = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {"C": [0.01, 0.1, 1, 10], "class_weight": [None, "balanced"]},
        "grid"
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"max_depth": [5, 10, 15, None], "min_samples_leaf": [1, 5, 10],
         "class_weight": [None, "balanced"]},
        "random"
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [100, 200], "max_depth": [10, 20, None],
         "min_samples_leaf": [1, 5], "class_weight": [None, "balanced"]},
        "random"
    ),
    "SVM": (
        SVC(probability=True, random_state=RANDOM_STATE),
        {"C": [0.1, 1, 10], "gamma": ["scale", 0.01], "kernel": ["rbf"]},
        "grid"
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"var_smoothing": [1e-9, 1e-8, 1e-7]},
        "grid"
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"],
         "metric": ["euclidean", "manhattan"]},
        "random"
    ),
    "ANN (MLP)": (
        MLPClassifier(max_iter=600, random_state=RANDOM_STATE, early_stopping=True),
        {"hidden_layer_sizes": [(50,), (100,), (50, 25)], "alpha": [0.0001, 0.001]},
        "grid"
    ),
}
if HAS_XGB:
    model_specs["XGBoost"] = (
        XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss"),
        {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1]},
        "random"
    )

print("\n" + "=" * 78)
print("STEP 5 — HYPERPARAMETER TUNING + 10-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 78)
print(f"{len(model_specs)} models to train. This will take a while — progress prints after each one.\n")

results = []
fitted_models = {}

for name, (estimator, grid, search_type) in model_specs.items():
    t0 = time.time()
    if search_type == "grid":
        search = GridSearchCV(estimator, grid, cv=cv5, scoring="f1_weighted", n_jobs=1)
    else:
        n_iter = min(10, int(np.prod([len(v) for v in grid.values()])))
        search = RandomizedSearchCV(estimator, grid, n_iter=n_iter, cv=cv5,
                                     scoring="f1_weighted", random_state=RANDOM_STATE, n_jobs=1)
    search.fit(X_train_scaled, y_train)
    best_model = search.best_estimator_
    fitted_models[name] = best_model

    # Robust 10-fold CV performance estimate on the tuned model (train data only)
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv10,
                                 scoring="f1_weighted", n_jobs=1)

    elapsed = time.time() - t0
    results.append({
        "Model": name,
        "Best_Hyperparameters": search.best_params_,
        "CV10_F1_Mean": round(cv_scores.mean(), 4),
        "CV10_F1_Std": round(cv_scores.std(), 4),
        "Fit_Time_Sec": round(elapsed, 1),
    })
    print(f"[{elapsed:6.1f}s] {name:<22} 10-fold CV weighted F1 = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"          Best hyperparameters: {search.best_params_}\n")

results_df = pd.DataFrame(results).sort_values("CV10_F1_Mean", ascending=False).reset_index(drop=True)

print("=" * 78)
print("MODEL COMPARISON — sorted by 10-fold CV weighted F1 (higher and more stable is better)")
print("=" * 78)
for _, row in results_df.iterrows():
    print(f"{row['Model']:<22} F1 = {row['CV10_F1_Mean']:.4f} ± {row['CV10_F1_Std']:.4f}   {row['Best_Hyperparameters']}")

results_df.to_csv("step5_hyperparameters_and_cv_scores.csv", index=False)
print("\nSaved full results table -> step5_hyperparameters_and_cv_scores.csv")
print("\nStep 5 complete. fitted_models (all 8 tuned models) and results_df are ready for Step 6 (validation-set evaluation and best-model selection).")

# ======================================================================
# STEP 6 — MODEL TRAINING + FEATURE IMPORTANCE
# ======================================================================

from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("\n" + "=" * 78)
print("STEP 6 — MODEL TRAINING + FEATURE IMPORTANCE")
print("=" * 78)

# ----------------------------------------------------------------------
# 1. Store feature names
# ----------------------------------------------------------------------

feature_names = X.columns.tolist()

print(f"\nTotal number of features: {len(feature_names)}")
print("Features used by the models:")
print(feature_names)


# ----------------------------------------------------------------------
# 2. Extract feature importance for every tuned model
# ----------------------------------------------------------------------

feature_importance_results = {}

for name, model in fitted_models.items():

    print("\n" + "-" * 78)
    print(f"MODEL: {name}")
    print("-" * 78)

    # --------------------------------------------------------------
    # Tree-based models
    # --------------------------------------------------------------

    if hasattr(model, "feature_importances_"):

        importance = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })

    # --------------------------------------------------------------
    # All other models
    # Use permutation importance on validation data
    # --------------------------------------------------------------

    else:

        perm_result = permutation_importance(
            model,
            X_val_scaled,
            y_val,
            scoring="f1_weighted",
            n_repeats=10,
            random_state=RANDOM_STATE,
            n_jobs=1
        )

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": perm_result.importances_mean
        })

    # --------------------------------------------------------------
    # Sort features by importance
    # --------------------------------------------------------------

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)

    feature_importance_results[name] = importance_df

    # --------------------------------------------------------------
    # Display top 10 features
    # --------------------------------------------------------------

    print("\nTop 10 Important Features:")

    print(
        importance_df.head(10).to_string(index=False)
    )


# ======================================================================
# 3. Save feature importance results for every model
# ======================================================================

with pd.ExcelWriter(
    "step6_feature_importance.xlsx",
    engine="openpyxl"
) as writer:

    for name, importance_df in feature_importance_results.items():

        # Excel sheet names cannot exceed 31 characters
        sheet_name = name[:31]

        importance_df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )


print("\nFeature importance results saved to:")
print("step6_feature_importance.xlsx")


# ======================================================================
# 4. Identify the best model from Step 5
# ======================================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = fitted_models[best_model_name]

print("\n" + "=" * 78)
print("BEST MODEL FROM STEP 5")
print("=" * 78)

print(f"Best model: {best_model_name}")

print(
    f"10-fold CV weighted F1: "
    f"{results_df.iloc[0]['CV10_F1_Mean']:.4f}"
)


# ======================================================================
# 5. Display feature importance of the best model
# ======================================================================

best_importance = feature_importance_results[best_model_name]

print("\n" + "=" * 78)
print(f"TOP FEATURES — {best_model_name}")
print("=" * 78)

print(
    best_importance.head(20).to_string(index=False)
)


# ======================================================================
# 6. Plot top 15 features of the best model
# ======================================================================

top_features = best_importance.head(15).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title(
    f"Top 15 Feature Importances — {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    "step6_best_model_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ======================================================================
# STEP 6 COMPLETE
# ======================================================================

print("\n" + "=" * 78)
print("STEP 6 COMPLETE")
print("=" * 78)

print("All tuned candidate models have been evaluated for feature importance.")

print(
    f"Best model identified from Step 5: "
    f"{best_model_name}"
)

print(
    "\nStep 6 output files:"
)

print("1. step6_feature_importance.xlsx")
print("2. step6_best_model_feature_importance.png")


# ======================================================================
# STEP 7 — MODEL OPTIMIZATION AND BEST MODEL SELECTION
# ======================================================================
#
# Methodology:
# 1. Compare all tuned models using VALIDATION weighted F1.
# 2. Select the best-performing model.
# 3. Compare the full-feature model with reduced models using
#    the top 20 and top 30 important features.
# 4. Keep the reduced model only when its performance is sufficiently
#    close to the full-feature model.
#
# IMPORTANT:
# The TEST set is NOT used anywhere in Step 7.
# It is reserved exclusively for Step 8.
# ======================================================================

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "=" * 78)
print("STEP 7 — MODEL OPTIMIZATION AND BEST MODEL SELECTION")
print("=" * 78)


# ----------------------------------------------------------------------
# 1. Evaluate every tuned model on the VALIDATION set
# ----------------------------------------------------------------------

validation_results = []

print("\n" + "-" * 78)
print("VALIDATION-SET MODEL COMPARISON")
print("-" * 78)

for name, model in fitted_models.items():

    # Predict validation data
    val_pred = model.predict(X_val_scaled)

    # Weighted F1
    val_f1 = f1_score(
        y_val,
        val_pred,
        average="weighted",
        zero_division=0
    )

    # Accuracy
    val_accuracy = accuracy_score(
        y_val,
        val_pred
    )

    # Weighted precision
    val_precision = precision_score(
        y_val,
        val_pred,
        average="weighted",
        zero_division=0
    )

    # Weighted recall
    val_recall = recall_score(
        y_val,
        val_pred,
        average="weighted",
        zero_division=0
    )

    validation_results.append({
        "Model": name,
        "Validation_Accuracy": round(val_accuracy, 4),
        "Validation_Precision_Weighted": round(val_precision, 4),
        "Validation_Recall_Weighted": round(val_recall, 4),
        "Validation_F1_Weighted": round(val_f1, 4)
    })

    print(
        f"{name:<22} "
        f"Accuracy = {val_accuracy:.4f}   "
        f"Weighted F1 = {val_f1:.4f}"
    )


# ----------------------------------------------------------------------
# 2. Sort models according to validation weighted F1
# ----------------------------------------------------------------------

validation_results_df = pd.DataFrame(
    validation_results
).sort_values(
    by="Validation_F1_Weighted",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 78)
print("MODELS SORTED BY VALIDATION WEIGHTED F1")
print("=" * 78)

print(
    validation_results_df.to_string(index=False)
)


# ----------------------------------------------------------------------
# 3. Select the best model
# ----------------------------------------------------------------------

best_model_name = validation_results_df.iloc[0]["Model"]

best_model = fitted_models[best_model_name]

best_validation_f1 = validation_results_df.iloc[0][
    "Validation_F1_Weighted"
]

print("\n" + "=" * 78)
print("BEST MODEL SELECTED")
print("=" * 78)

print(f"Best Model               : {best_model_name}")
print(f"Validation Weighted F1   : {best_validation_f1:.4f}")


# ----------------------------------------------------------------------
# 4. Get feature importance of the selected model
# ----------------------------------------------------------------------

best_importance = feature_importance_results[
    best_model_name
].copy()

best_importance = best_importance.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)


print("\n" + "-" * 78)
print("TOP IMPORTANT FEATURES")
print("-" * 78)

print(
    best_importance.head(30).to_string(index=False)
)


# ----------------------------------------------------------------------
# 5. Define top 20 and top 30 feature sets
# ----------------------------------------------------------------------

total_features = len(feature_names)

top20_count = min(20, total_features)
top30_count = min(30, total_features)

top20_features = best_importance.head(
    top20_count
)["Feature"].tolist()

top30_features = best_importance.head(
    top30_count
)["Feature"].tolist()


print("\nTop 20 features selected :", len(top20_features))
print("Top 30 features selected :", len(top30_features))


# ----------------------------------------------------------------------
# 6. Create column-index mappings
# ----------------------------------------------------------------------

feature_index = {
    feature: index
    for index, feature in enumerate(feature_names)
}

top20_indices = [
    feature_index[f]
    for f in top20_features
]

top30_indices = [
    feature_index[f]
    for f in top30_features
]


# ----------------------------------------------------------------------
# 7. Prepare reduced feature datasets
# ----------------------------------------------------------------------

X_train_top20 = X_train_scaled[:, top20_indices]
X_val_top20 = X_val_scaled[:, top20_indices]

X_train_top30 = X_train_scaled[:, top30_indices]
X_val_top30 = X_val_scaled[:, top30_indices]


# ----------------------------------------------------------------------
# 8. Evaluate FULL-FEATURE model on validation data
# ----------------------------------------------------------------------

full_val_pred = best_model.predict(
    X_val_scaled
)

full_val_f1 = f1_score(
    y_val,
    full_val_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 78)
print("FULL-FEATURE MODEL")
print("=" * 78)

print(f"Features used          : {len(feature_names)}")
print(f"Validation Weighted F1  : {full_val_f1:.4f}")


# ----------------------------------------------------------------------
# 9. Train/evaluate TOP-20 feature model
# ----------------------------------------------------------------------

top20_model = clone(best_model)

top20_model.fit(
    X_train_top20,
    y_train
)

top20_val_pred = top20_model.predict(
    X_val_top20
)

top20_val_f1 = f1_score(
    y_val,
    top20_val_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 78)
print("TOP-20 FEATURE MODEL")
print("=" * 78)

print(f"Features used          : {len(top20_features)}")
print(f"Validation Weighted F1  : {top20_val_f1:.4f}")


# ----------------------------------------------------------------------
# 10. Train/evaluate TOP-30 feature model
# ----------------------------------------------------------------------

top30_model = clone(best_model)

top30_model.fit(
    X_train_top30,
    y_train
)

top30_val_pred = top30_model.predict(
    X_val_top30
)

top30_val_f1 = f1_score(
    y_val,
    top30_val_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 78)
print("TOP-30 FEATURE MODEL")
print("=" * 78)

print(f"Features used          : {len(top30_features)}")
print(f"Validation Weighted F1  : {top30_val_f1:.4f}")


# ----------------------------------------------------------------------
# 11. Compare full vs reduced models
# ----------------------------------------------------------------------
#
# We allow the reduced model to be up to 0.01 F1 below the full model.
# This prevents selecting a much weaker reduced model simply because
# it uses fewer variables.
# ----------------------------------------------------------------------

F1_TOLERANCE = 0.01

reduced_results = pd.DataFrame({
    "Feature_Set": [
        "Full Features",
        "Top 20 Features",
        "Top 30 Features"
    ],
    "Number_of_Features": [
        len(feature_names),
        len(top20_features),
        len(top30_features)
    ],
    "Validation_Weighted_F1": [
        full_val_f1,
        top20_val_f1,
        top30_val_f1
    ]
})

print("\n" + "=" * 78)
print("FULL VS REDUCED FEATURE COMPARISON")
print("=" * 78)

print(
    reduced_results.to_string(index=False)
)


# ----------------------------------------------------------------------
# 12. Select final feature set
# ----------------------------------------------------------------------

# Candidate reduced models whose F1 is close enough to full model
acceptable_reduced_models = []

if top20_val_f1 >= full_val_f1 - F1_TOLERANCE:
    acceptable_reduced_models.append(
        ("Top 20 Features", top20_val_f1)
    )

if top30_val_f1 >= full_val_f1 - F1_TOLERANCE:
    acceptable_reduced_models.append(
        ("Top 30 Features", top30_val_f1)
    )


# Choose the smallest acceptable feature set
if len(acceptable_reduced_models) > 0:

    acceptable_reduced_models.sort(
        key=lambda x: (
            20 if x[0] == "Top 20 Features" else 30,
            -x[1]
        )
    )

    selected_feature_set = acceptable_reduced_models[0][0]

    if selected_feature_set == "Top 20 Features":

        selected_model = top20_model
        selected_features = top20_features
        selected_feature_indices = top20_indices

    else:

        selected_model = top30_model
        selected_features = top30_features
        selected_feature_indices = top30_indices

else:

    selected_feature_set = "Full Features"

    selected_model = best_model
    selected_features = feature_names
    selected_feature_indices = list(
        range(len(feature_names))
    )


# ----------------------------------------------------------------------
# 13. Display final selection
# ----------------------------------------------------------------------

selected_validation_f1 = max(
    [
        full_val_f1 if selected_feature_set == "Full Features" else -1,
        top20_val_f1 if selected_feature_set == "Top 20 Features" else -1,
        top30_val_f1 if selected_feature_set == "Top 30 Features" else -1
    ]
)

print("\n" + "=" * 78)
print("FINAL MODEL SELECTION")
print("=" * 78)

print(f"Base model              : {best_model_name}")
print(f"Selected feature set    : {selected_feature_set}")
print(f"Number of features      : {len(selected_features)}")
print(f"Validation Weighted F1  : {selected_validation_f1:.4f}")


# ----------------------------------------------------------------------
# 14. Save Step 7 results
# ----------------------------------------------------------------------

validation_results_df.to_excel(
    "step7_model_validation_comparison.xlsx",
    index=False
)

reduced_results.to_excel(
    "step7_feature_reduction_comparison.xlsx",
    index=False
)

pd.DataFrame({
    "Selected_Model": [best_model_name],
    "Selected_Feature_Set": [selected_feature_set],
    "Number_of_Selected_Features": [len(selected_features)],
    "Validation_Weighted_F1": [selected_validation_f1]
}).to_excel(
    "step7_final_model_selection.xlsx",
    index=False
)

pd.DataFrame({
    "Feature": selected_features
}).to_excel(
    "step7_selected_features.xlsx",
    index=False
)


# ----------------------------------------------------------------------
# 15. Save selected model object temporarily for Step 8
# ----------------------------------------------------------------------

step7_selection = {
    "model_name": best_model_name,
    "feature_set": selected_feature_set,
    "selected_model": selected_model,
    "selected_features": selected_features,
    "selected_feature_indices": selected_feature_indices,
    "validation_f1": selected_validation_f1
}


print("\nStep 7 output files:")
print("1. step7_model_validation_comparison.xlsx")
print("2. step7_feature_reduction_comparison.xlsx")
print("3. step7_final_model_selection.xlsx")
print("4. step7_selected_features.xlsx")

print("\n" + "=" * 78)
print("STEP 7 COMPLETE")
print("=" * 78)


# ======================================================================
# STEP 8 — FINAL MODEL EVALUATION AND DOCUMENTATION
# ======================================================================
#
# IMPORTANT:
# The held-out TEST set is used ONLY here.
#
# Step 7 selected:
#   1. Best model
#   2. Best feature set
#
# Step 8 evaluates that selected model on X_test / y_test.
# ======================================================================

import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


print("\n" + "=" * 78)
print("STEP 8 — FINAL HELD-OUT TEST SET EVALUATION")
print("=" * 78)


# ----------------------------------------------------------------------
# 1. Confirm the selected model from Step 7
# ----------------------------------------------------------------------

final_model = step7_selection["selected_model"]
final_model_name = step7_selection["model_name"]
final_feature_set = step7_selection["feature_set"]
final_features = step7_selection["selected_features"]
final_feature_indices = step7_selection["selected_feature_indices"]

print("\nSelected model:")
print(final_model_name)

print("\nSelected feature set:")
print(final_feature_set)

print("\nNumber of selected features:")
print(len(final_features))


# ----------------------------------------------------------------------
# 2. Prepare TEST data using the selected feature set
# ----------------------------------------------------------------------
#
# IMPORTANT:
# X_test_scaled was already generated in Step 4 using the scaler
# fitted ONLY on X_train.
#
# We do not fit another scaler here.
# ----------------------------------------------------------------------

X_test_selected = X_test_scaled[
    :,
    final_feature_indices
]


# ----------------------------------------------------------------------
# 3. FINAL TEST PREDICTION
# ----------------------------------------------------------------------

print("\n" + "-" * 78)
print("FINAL TEST SET PREDICTION")
print("-" * 78)

test_predictions = final_model.predict(
    X_test_selected
)


# ----------------------------------------------------------------------
# 4. Prediction probabilities
# ----------------------------------------------------------------------

try:

    test_probabilities = final_model.predict_proba(
        X_test_selected
    )

    roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
        multi_class="ovr",
        average="weighted"
    )

except Exception as e:

    test_probabilities = None
    roc_auc = np.nan

    print("\nROC-AUC could not be calculated:")
    print(e)


# ----------------------------------------------------------------------
# 5. Accuracy
# ----------------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    test_predictions
)


# ----------------------------------------------------------------------
# 6. Precision
# ----------------------------------------------------------------------

precision = precision_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)


# ----------------------------------------------------------------------
# 7. Recall
# ----------------------------------------------------------------------

recall = recall_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)


# ----------------------------------------------------------------------
# 8. Weighted F1
# ----------------------------------------------------------------------

f1 = f1_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)


# ----------------------------------------------------------------------
# 9. Matthews Correlation Coefficient
# ----------------------------------------------------------------------

mcc = matthews_corrcoef(
    y_test,
    test_predictions
)


# ----------------------------------------------------------------------
# 10. Display FINAL TEST METRICS
# ----------------------------------------------------------------------

print("\n" + "=" * 78)
print("FINAL TEST SET PERFORMANCE")
print("=" * 78)

print(f"Model                         : {final_model_name}")
print(f"Feature Set                   : {final_feature_set}")
print(f"Number of Features            : {len(final_features)}")

print("\nPerformance Metrics")
print("-" * 50)

print(f"Accuracy                      : {accuracy:.4f}")
print(f"Precision (Weighted)          : {precision:.4f}")
print(f"Recall (Weighted)             : {recall:.4f}")
print(f"F1-Score (Weighted)           : {f1:.4f}")

if not np.isnan(roc_auc):

    print(
        f"ROC-AUC (Weighted, OvR)      : {roc_auc:.4f}"
    )

else:

    print(
        "ROC-AUC (Weighted, OvR)      : Not Available"
    )

print(
    f"Matthews Correlation Coeff.   : {mcc:.4f}"
)


# ----------------------------------------------------------------------
# 11. Classification report
# ----------------------------------------------------------------------

class_names = [
    "Rejected",
    "Pending",
    "Manual Review",
    "Approved"
]

print("\n" + "=" * 78)
print("CLASSIFICATION REPORT")
print("=" * 78)

classification_report_text = classification_report(
    y_test,
    test_predictions,
    labels=[0, 1, 2, 3],
    target_names=class_names,
    zero_division=0
)

print(classification_report_text)


# ----------------------------------------------------------------------
# 12. Confusion Matrix
# ----------------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1, 2, 3]
)

print("\n" + "=" * 78)
print("CONFUSION MATRIX")
print("=" * 78)

print(cm)


# ----------------------------------------------------------------------
# 13. Plot confusion matrix
# ----------------------------------------------------------------------

plt.figure(
    figsize=(9, 7)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.title(
    f"Final Test Confusion Matrix — {final_model_name}"
)

import os

plt.tight_layout()

# Get the folder where this Python file is located
output_folder = os.path.dirname(os.path.abspath(__file__))

# Create the complete output path
output_file = os.path.join(
    output_folder,
    "step8_final_confusion_matrix.png"
)

# Save the confusion matrix
plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
    format="png"
)

print("\nConfusion matrix image saved successfully:")
print(output_file)

plt.show()
plt.close()

plt.show()


# ----------------------------------------------------------------------
# 14. Save final performance metrics
# ----------------------------------------------------------------------

final_performance = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision (Weighted)",
        "Recall (Weighted)",
        "F1-Score (Weighted)",
        "ROC-AUC (Weighted OvR)",
        "Matthews Correlation Coefficient"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        mcc
    ]
})


final_performance.to_excel(
    "step8_final_test_performance.xlsx",
    index=False
)


# ----------------------------------------------------------------------
# 15. Save confusion matrix
# ----------------------------------------------------------------------

confusion_matrix_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

confusion_matrix_df.to_excel(
    "step8_final_confusion_matrix.xlsx"
)


# ----------------------------------------------------------------------
# 16. Save selected feature list
# ----------------------------------------------------------------------

selected_features_df = pd.DataFrame({
    "Feature_Number": range(
        1,
        len(final_features) + 1
    ),
    "Feature": final_features
})

selected_features_df.to_excel(
    "step8_final_selected_features.xlsx",
    index=False
)


# ----------------------------------------------------------------------
# 17. Save final model
# ----------------------------------------------------------------------

joblib.dump(
    final_model,
    "loan_eligibility_final_model.pkl"
)


# ----------------------------------------------------------------------
# 18. Save scaler
# ----------------------------------------------------------------------
#
# This is the scaler fitted in Step 4.
# It was fitted ONLY on X_train.
# ----------------------------------------------------------------------

joblib.dump(
    scaler,
    "loan_eligibility_scaler.pkl"
)


# ----------------------------------------------------------------------
# 19. Save model documentation
# ----------------------------------------------------------------------

documentation = {
    "Project": "AI-Powered Loan Eligibility Assessment System",

    "Final_Model": final_model_name,

    "Feature_Set": final_feature_set,

    "Number_of_Features": len(final_features),

    "Train_Validation_Test_Split": "70% / 15% / 15%",

    "Random_State": RANDOM_STATE,

    "Cross_Validation": "5-fold tuning + 10-fold stratified CV",

    "Primary_Model_Selection_Metric":
        "Weighted F1",

    "Validation_Weighted_F1":
        float(step7_selection["validation_f1"]),

    "Test_Accuracy":
        float(accuracy),

    "Test_Precision_Weighted":
        float(precision),

    "Test_Recall_Weighted":
        float(recall),

    "Test_F1_Weighted":
        float(f1),

    "Test_ROC_AUC_Weighted":
        None if np.isnan(roc_auc) else float(roc_auc),

    "Test_MCC":
        float(mcc),

    "Selected_Features":
        final_features,

    "Test_Set_Used_Only_In_Step_8":
        True
}


# Convert any NumPy values into normal Python values
documentation_json = json.dumps(
    documentation,
    indent=4
)


with open(
    "step8_model_documentation.json",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        documentation_json
    )


# ----------------------------------------------------------------------
# 20. Save a human-readable documentation file
# ----------------------------------------------------------------------

with open(
    "step8_model_documentation.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AI-POWERED LOAN ELIGIBILITY ASSESSMENT SYSTEM\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        f"Final Model: {final_model_name}\n"
    )

    file.write(
        f"Feature Set: {final_feature_set}\n"
    )

    file.write(
        f"Number of Features: {len(final_features)}\n"
    )

    file.write(
        "Train / Validation / Test Split: 70% / 15% / 15%\n"
    )

    file.write(
        "Cross-Validation: 5-fold tuning + 10-fold stratified CV\n"
    )

    file.write(
        "Primary Metric: Weighted F1\n\n"
    )

    file.write(
        "FINAL TEST PERFORMANCE\n"
    )

    file.write(
        "-" * 50 + "\n"
    )

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Precision (Weighted): {precision:.4f}\n"
    )

    file.write(
        f"Recall (Weighted): {recall:.4f}\n"
    )

    file.write(
        f"F1-Score (Weighted): {f1:.4f}\n"
    )

    if not np.isnan(roc_auc):

        file.write(
            f"ROC-AUC (Weighted OvR): {roc_auc:.4f}\n"
        )

    else:

        file.write(
            "ROC-AUC (Weighted OvR): Not Available\n"
        )

    file.write(
        f"Matthews Correlation Coefficient: {mcc:.4f}\n"
    )

    file.write(
        "\nSELECTED FEATURES\n"
    )

    file.write(
        "-" * 50 + "\n"
    )

    for i, feature in enumerate(
        final_features,
        start=1
    ):

        file.write(
            f"{i}. {feature}\n"
        )


# ----------------------------------------------------------------------
# 21. Final completion message
# ----------------------------------------------------------------------

print("\n" + "=" * 78)
print("STEP 8 COMPLETE")
print("=" * 78)

print("\nFinal Model:")
print(final_model_name)

print("\nFinal Feature Set:")
print(final_feature_set)

print("\nNumber of Features:")
print(len(final_features))

print("\nFINAL TEST METRICS")
print("-" * 50)

print(f"Accuracy                    : {accuracy:.4f}")
print(f"Precision (Weighted)        : {precision:.4f}")
print(f"Recall (Weighted)           : {recall:.4f}")
print(f"F1-Score (Weighted)         : {f1:.4f}")

if not np.isnan(roc_auc):
    print(f"ROC-AUC (Weighted)          : {roc_auc:.4f}")
else:
    print("ROC-AUC (Weighted)          : Not Available")

print(f"MCC                         : {mcc:.4f}")

print("\nSaved files:")
print("1. loan_eligibility_final_model.pkl")
print("2. loan_eligibility_scaler.pkl")
print("3. step8_final_test_performance.xlsx")
print("4. step8_final_confusion_matrix.xlsx")
print("5. step8_final_confusion_matrix.png")
print("6. step8_final_selected_features.xlsx")
print("7. step8_model_documentation.json")
print("8. step8_model_documentation.txt")

print("\n" + "=" * 78)
print("PHASE 4 STEPS 1–8 COMPLETE")
print("=" * 78)

#PHASE 5 :TRAINING THE MACHINE LEARNING MODEL

print("=" * 78)
print("\n Phase 5 implementation: MACHINE LEARNING MODELLING ")
print("=" * 78)

# ======================================================================
# Phase 5 implementation: MACHINE LEARNING MODELLING
# ======================================================================

y_pred = test_predictions          # <-- add this
y_proba = test_probabilities       # <-- add this


protected_groups = {
    "Gender": ["Gender_Female", "Gender_Male", "Gender_Other"],
    "Marital_Status": ["Marital_Status_Divorced", "Marital_Status_Married",
                        "Marital_Status_Single", "Marital_Status_Widowed"],
}

fairness_results = []
test_idx = X_test.index  # or however you track test rows back to df

for group_name, dummy_cols in protected_groups.items():
    for col in dummy_cols:
        mask = df.loc[test_idx, col] == 1
        if mask.sum() < 20:
            continue
        y_true_g = y_test[mask.values]
        y_pred_g = y_pred[mask.values]

        approval_rate = (y_pred_g == APPROVED_CODE).mean()
        recall_g = recall_score(y_true_g, y_pred_g, average="weighted")
        f1_g = f1_score(y_true_g, y_pred_g, average="weighted")

        fairness_results.append({
            "Attribute": group_name, "Category": col,
            "N": mask.sum(), "Approval_Rate": approval_rate,
            "Recall_Weighted": recall_g, "F1_Weighted": f1_g
        })

fairness_df = pd.DataFrame(fairness_results)

max_rate = fairness_df.groupby("Attribute")["Approval_Rate"].transform("max")
fairness_df["Disparate_Impact_Ratio"] = fairness_df["Approval_Rate"] / max_rate
fairness_df["Flag_Below_0.8"] = fairness_df["Disparate_Impact_Ratio"] < 0.8
fairness_df.to_excel("step5_rbi_fairness_audit.xlsx", index=False)

protected_cols = sum(protected_groups.values(), [])
flagged = best_importance[best_importance["Feature"].isin(protected_cols)]
flagged.to_excel("step5_protected_attribute_check.xlsx", index=False)

import shap

# ======================================================================
# SHAP EXPLAINABILITY ANALYSIS
# ======================================================================

import shap

print("\n" + "=" * 78)
print("SHAP EXPLAINABILITY ANALYSIS")
print("=" * 78)

# IMPORTANT:
# final_model was trained using the selected feature set, NOT all
# original scaled features.
#
# Therefore SHAP must receive the same feature columns used by
# final_model.

X_train_selected = X_train_scaled[:, final_feature_indices]
X_test_selected = X_test_scaled[:, final_feature_indices]

print("Total original features :", X_train_scaled.shape[1])
print("Features used by model  :", X_train_selected.shape[1])
print("SHAP test features      :", X_test_selected.shape[1])

# Safety check
if X_train_selected.shape[1] != X_test_selected.shape[1]:
    raise ValueError(
        "SHAP feature mismatch: training and test feature counts differ."
    )

if X_train_selected.shape[1] != len(final_features):
    raise ValueError(
        f"Feature-name mismatch: model uses {X_train_selected.shape[1]} "
        f"features but final_features contains {len(final_features)} names."
    )

# ======================================================================
# SHAP EXPLAINABILITY ANALYSIS
# ======================================================================

import shap

print("\n" + "=" * 78)
print("SHAP EXPLAINABILITY ANALYSIS")
print("=" * 78)

# ---------------------------------------------------------------
# Use EXACTLY the features used to train the final model
# ---------------------------------------------------------------

X_train_selected = X_train_scaled[:, final_feature_indices]
X_test_selected = X_test_scaled[:, final_feature_indices]

print("Original number of features :", X_train_scaled.shape[1])
print("Selected model features    :", X_train_selected.shape[1])
print("SHAP test features         :", X_test_selected.shape[1])
print("Feature names              :", len(final_features))

# Safety check
if X_train_selected.shape[1] != len(final_features):
    raise ValueError(
        "Mismatch between selected feature matrix and final feature names."
    )

if X_test_selected.shape[1] != len(final_features):
    raise ValueError(
        "Mismatch between test feature matrix and final feature names."
    )

# ---------------------------------------------------------------
# IMPORTANT:
# Use TreeExplainer with tree_path_dependent.
#
# This avoids the categorical-split error produced by the generic
# shap.Explainer call for the XGBoost model.
# ---------------------------------------------------------------

explainer = shap.TreeExplainer(
    final_model,
    feature_perturbation="tree_path_dependent"
)

# Calculate SHAP values
shap_values = explainer.shap_values(
    X_test_selected
)

print("\nSHAP calculation completed successfully.")

# ---------------------------------------------------------------
# Convert SHAP output to a consistent NumPy array
# ---------------------------------------------------------------

if isinstance(shap_values, list):

    # Older SHAP versions may return a list for multiclass models
    shap_array = np.stack(shap_values, axis=-1)

else:

    shap_array = np.asarray(shap_values)

print("SHAP array shape:", shap_array.shape)

# ---------------------------------------------------------------
# Use FINAL MODEL predictions
# ---------------------------------------------------------------

final_predictions = final_model.predict(
    X_test_selected
)

# ---------------------------------------------------------------
# Generate adverse-action reason codes
# ---------------------------------------------------------------

rejected_mask = (
    final_predictions != APPROVED_CODE
)

reason_codes = []

for i in np.where(rejected_mask)[0]:

    # -----------------------------------------------------------
    # Multiclass SHAP:
    # shape = samples × features × classes
    # -----------------------------------------------------------

    if shap_array.ndim == 3:

        predicted_class = final_predictions[i]

        class_position = list(
            final_model.classes_
        ).index(predicted_class)

        contributions = shap_array[
            i,
            :,
            class_position
        ]

    # -----------------------------------------------------------
    # Binary / single-output SHAP:
    # shape = samples × features
    # -----------------------------------------------------------

    elif shap_array.ndim == 2:

        contributions = shap_array[i]

    else:

        raise ValueError(
            f"Unexpected SHAP output shape: {shap_array.shape}"
        )

    # -----------------------------------------------------------
    # Make sure number of SHAP values equals number of features
    # -----------------------------------------------------------

    if len(contributions) != len(final_features):

        raise ValueError(
            f"SHAP feature mismatch: "
            f"{len(contributions)} SHAP values but "
            f"{len(final_features)} feature names."
        )

    # -----------------------------------------------------------
    # Find most influential features
    # -----------------------------------------------------------

    contrib = pd.Series(
        contributions,
        index=final_features
    )

    top_reasons = (
        contrib.abs()
        .sort_values(ascending=False)
        .head(3)
        .index
        .tolist()
    )

    reason_codes.append({
        "Applicant_Index": int(i),
        "Predicted_Class": final_predictions[i],
        "Top_Reason_1": (
            top_reasons[0]
            if len(top_reasons) > 0 else ""
        ),
        "Top_Reason_2": (
            top_reasons[1]
            if len(top_reasons) > 1 else ""
        ),
        "Top_Reason_3": (
            top_reasons[2]
            if len(top_reasons) > 2 else ""
        )
    })

# ---------------------------------------------------------------
# Save results
# ---------------------------------------------------------------

reason_codes_df = pd.DataFrame(
    reason_codes
)

reason_codes_df.to_excel(
    "step5_adverse_action_reasons.xlsx",
    index=False
)

print("\nSHAP reason-code analysis completed.")
print(
    "Saved: step5_adverse_action_reasons.xlsx"
)

print("=" * 78)

print("\nSHAP reason-code analysis saved successfully:")
print("step5_adverse_action_reasons.xlsx")

print("=" * 78)

def decision_band(prob_approve, low=0.40, high=0.60):
    if prob_approve >= high: return "Approve"
    elif prob_approve <= low: return "Reject"
    else: return "Manual Review"

approve_idx = list(final_model.classes_).index(APPROVED_CODE)
prob_approve = y_proba[:, approve_idx]
decisions = pd.Series(prob_approve).apply(decision_band)
print(decisions.value_counts(normalize=True))

compliance_summary = df.loc[test_idx, [
    "KFS_Provided_to_Borrower", "Explicit_Data_Consent_Obtained",
    "Video_KYC_Completed", "Data_Localisation_Compliant",
    "Cooling_Off_Period_Days"
]].agg(["mean", "min"])
compliance_summary.to_excel("step5_process_compliance_summary.xlsx")

for risk_col in ["Applicant_Risk_Category_Low_Risk",
                  "Applicant_Risk_Category_Medium_Risk",
                  "Applicant_Risk_Category_High_Risk"]:
    mask = df.loc[test_idx, risk_col] == 1
    if mask.sum() > 10:
        print(risk_col, "F1:", f1_score(y_test[mask.values], y_pred[mask.values], average="weighted"))

