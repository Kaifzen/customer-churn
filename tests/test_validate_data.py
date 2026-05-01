import pandas as pd

from src.utils.validate_data import validate_telco_data


def _valid_row() -> dict:
    return {
        "customerID": "0001-A",
        "gender": "Female",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "InternetService": "Fiber optic",
        "Contract": "Month-to-month",
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 846.0,
    }


def test_validate_telco_data_accepts_valid_input():
    df = pd.DataFrame([_valid_row()])

    is_valid, failures = validate_telco_data(df)

    assert is_valid is True
    assert failures == []


def test_validate_telco_data_rejects_invalid_values():
    bad = _valid_row()
    bad["gender"] = "Other"
    bad["tenure"] = -1

    df = pd.DataFrame([bad])

    is_valid, failures = validate_telco_data(df)

    assert is_valid is False
    assert "expect_column_values_to_be_in_set:gender" in failures
    assert "expect_column_values_to_be_between:tenure" in failures
