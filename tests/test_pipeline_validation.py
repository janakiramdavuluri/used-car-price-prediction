import pandas as pd
import pytest

from pipelines.run_lab7_pipeline import validate_data


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": "Example Car",
                "year": 2018,
                "selling_price": 250000,
                "km_driven": 40000,
                "fuel": "Petrol",
                "seller_type": "Individual",
                "transmission": "Manual",
                "owner": "First Owner",
            }
        ]
    )


def test_valid_data_is_deduplicated():
    data = pd.concat([valid_frame(), valid_frame()], ignore_index=True)
    result = validate_data(data, list(valid_frame().columns))
    assert len(result) == 1


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("selling_price", 0, "selling_price must be positive"),
        ("km_driven", -1, "km_driven cannot be negative"),
        ("year", "not-a-year", "contains non-numeric values"),
    ],
)
def test_invalid_values_stop_pipeline(column, value, message):
    data = valid_frame()
    data[column] = data[column].astype(object)
    data.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        validate_data(data, list(valid_frame().columns))


def test_missing_column_stops_pipeline():
    data = valid_frame().drop(columns=["owner"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(data, list(valid_frame().columns))