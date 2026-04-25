from typing import List, Tuple

import pandas as pd

try:
    import great_expectations as ge
except Exception:  # pragma: no cover - defensive import fallback
    ge = None


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    
    """
    print("Starting data validation with Great Expectations compatibility checks...")

    # GE <1.0 path: keep legacy behavior if available.
    if ge is not None and hasattr(ge, "dataset") and hasattr(ge.dataset, "PandasDataset"):
        ge_df = ge.dataset.PandasDataset(df)
        ge_df.expect_column_to_exist("customerID")
        ge_df.expect_column_values_to_not_be_null("customerID")
        ge_df.expect_column_to_exist("gender")
        ge_df.expect_column_to_exist("Partner")
        ge_df.expect_column_to_exist("Dependents")
        ge_df.expect_column_to_exist("PhoneService")
        ge_df.expect_column_to_exist("InternetService")
        ge_df.expect_column_to_exist("Contract")
        ge_df.expect_column_to_exist("tenure")
        ge_df.expect_column_to_exist("MonthlyCharges")
        ge_df.expect_column_to_exist("TotalCharges")
        ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
        ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
        ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
        ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])
        ge_df.expect_column_values_to_be_in_set(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        ge_df.expect_column_values_to_be_in_set(
            "InternetService", ["DSL", "Fiber optic", "No"]
        )
        ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
        ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)
        ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)
        ge_df.expect_column_values_to_not_be_null("tenure")
        ge_df.expect_column_values_to_not_be_null("MonthlyCharges")
        ge_df.expect_column_pair_values_A_to_be_greater_than_B(
            column_A="TotalCharges",
            column_B="MonthlyCharges",
            or_equal=True,
            mostly=0.95,
        )

        results = ge_df.validate()
        failed_expectations = [
            r["expectation_config"]["expectation_type"]
            for r in results["results"]
            if not r["success"]
        ]
        return bool(results["success"]), failed_expectations

    # GE 1.x path: perform equivalent dataframe checks directly.
    failed_expectations: List[str] = []

    required_columns = [
        "customerID",
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "InternetService",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        for col in missing:
            failed_expectations.append(f"expect_column_to_exist:{col}")
        print(f"Data validation FAILED: missing required columns -> {missing}")
        return False, failed_expectations

    def _all_values_in_set(column: str, allowed: set[str]) -> bool:
        values = set(df[column].dropna().astype(str).unique())
        return values.issubset(allowed)

    if df["customerID"].isna().any():
        failed_expectations.append("expect_column_values_to_not_be_null:customerID")

    if not _all_values_in_set("gender", {"Male", "Female"}):
        failed_expectations.append("expect_column_values_to_be_in_set:gender")
    if not _all_values_in_set("Partner", {"Yes", "No"}):
        failed_expectations.append("expect_column_values_to_be_in_set:Partner")
    if not _all_values_in_set("Dependents", {"Yes", "No"}):
        failed_expectations.append("expect_column_values_to_be_in_set:Dependents")
    if not _all_values_in_set("PhoneService", {"Yes", "No"}):
        failed_expectations.append("expect_column_values_to_be_in_set:PhoneService")
    if not _all_values_in_set("Contract", {"Month-to-month", "One year", "Two year"}):
        failed_expectations.append("expect_column_values_to_be_in_set:Contract")
    if not _all_values_in_set("InternetService", {"DSL", "Fiber optic", "No"}):
        failed_expectations.append("expect_column_values_to_be_in_set:InternetService")

    tenure = pd.to_numeric(df["tenure"], errors="coerce")
    monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if tenure.isna().any():
        failed_expectations.append("expect_column_values_to_not_be_null:tenure")
    if monthly.isna().any():
        failed_expectations.append("expect_column_values_to_not_be_null:MonthlyCharges")
    if (tenure < 0).any() or (tenure > 120).any():
        failed_expectations.append("expect_column_values_to_be_between:tenure")
    if (monthly < 0).any() or (monthly > 200).any():
        failed_expectations.append("expect_column_values_to_be_between:MonthlyCharges")
    if (total < 0).any():
        failed_expectations.append("expect_column_values_to_be_between:TotalCharges")

    comparable = total.notna() & monthly.notna()
    if comparable.any():
        mostly_ratio = (total[comparable] >= monthly[comparable]).mean()
        if mostly_ratio < 0.95:
            failed_expectations.append(
                "expect_column_pair_values_A_to_be_greater_than_B:TotalCharges>=MonthlyCharges"
            )

    is_valid = len(failed_expectations) == 0
    if is_valid:
        print("Data validation PASSED")
    else:
        print(f"Data validation FAILED: {failed_expectations}")
    return is_valid, failed_expectations