from __future__ import annotations

from pathlib import Path
from typing import Any, List, Set

import pytest

from edm2023 import dataset


@pytest.fixture(scope="module")
def edm2023_raw_dataset(raw_data_dir: Path) -> dataset.RawData:
    """Fixture for raw dataset"""
    return dataset.RawData.from_dir(raw_data_dir)


@pytest.fixture(scope="module")
def edm2023_processed_dataset(processed_data_dir: Path) -> dataset.ProcessedData:
    """Fixture for processed dataset"""
    return dataset.ProcessedData.from_dir(processed_data_dir)


@pytest.fixture(scope="module")
def unit_test_log_id_to_assignment_log_id_relation(edm2023_raw_dataset):
    """Fixture for test log id to assignment log id relation"""
    return dataset.TestLogIdToAssignmentLogIdRelation(
        edm2023_raw_dataset.assignment_relationships
    )


@pytest.mark.parametrize(
    "source_value, target_values",
    [
        (
            "7FGC8P0F1",
            {
                "V6YXT3UG",
                "1M8Z2KJ9PJ",
                "25RBWECCLY",
                "2AMHPZ86KV",
                "J0AMKEBD0",
                "J77JUXJI4",
                "2MMXVQCU24",
            },
        ),
        (
            "15KQFID5U5",
            {
                "1TFFYMT814",
                "9FNS9DC8N",
                "ZS5XLJIDC",
                "NASZ4E1Y9",
                "24U3NTWY7W",
                "4TTYQ9RR9",
                "1NMNY59NDO",
            },
        ),
    ],
)
def test_unit_test_log_id_to_assignment_log_id_relation(
    unit_test_log_id_to_assignment_log_id_relation, source_value, target_values
):
    result = unit_test_log_id_to_assignment_log_id_relation.get_target_attribute_values(
        source_value
    )

    assert target_values == result


@pytest.mark.parametrize(
    "dataframe_name, local_attribute_name, local_attribute_vals, pk_col, expected_pk_col_values",
    [
        (
            "edm_cup_2023_in_unit_assignment_scorable_actions",
            "unit_test_assignment_log_id",
            {"1WNMP4AOTN", "DVCHC37B2", "5WV1S0A43", "HKZSRA380"},
            ["unit_test_assignment_log_id", "problem_id"],
            [
                ["1WNMP4AOTN", "28UPV22XPX"],
                ["DVCHC37B2", "2GR24FQAEG"],
                ["5WV1S0A43", "2MHGD8MJE4"],
                ["HKZSRA380", "1GZL3VNLOF"],
            ],
        ),
    ],
)
def test_filter_dataset_by_local_attributes(
    edm2023_processed_dataset,
    dataframe_name,
    local_attribute_name,
    local_attribute_vals,
    pk_col,
    expected_pk_col_values,
):
    df = getattr(edm2023_processed_dataset, dataframe_name)

    filtered = dataset.DataFrameFilter.filter_by_local_attribute(
        df, local_attribute_name, local_attribute_vals
    )

    assert sorted(filtered[pk_col].values.tolist()) == sorted(expected_pk_col_values)


@pytest.mark.skip("filter_dataset_by_foreign_attributes is not needed at the moment")
@pytest.mark.parametrize(
    "dataframe_name, foreign_attribute_name, local_attribute_name, foreign_attribute_values, "
    "foreign_attribute_to_local_attribute_relation, pk_col, expected_pk_col_values",
    [
        (
            "problem_details",
            "unit_test_assignment_log_id",
            "problem_id",
            {...},
            ...,
            "problem_id",
            [...],
        )
    ],
)
def test_filter_dataset_by_foreign_attributes(
    edm2023_raw_dataset,
    dataframe_name: str,
    foreign_attribute_name: str,
    local_attribute_name: str,
    foreign_attribute_values: Set[Any],
    foreign_attribute_to_local_attribute_relation: dataset.IRelation,
    pk_col: str,
    expected_pk_col_values: List[Any],
):
    df = getattr(edm2023_raw_dataset, dataframe_name)
    filtered = dataset.DataFrameFilter.filter_by_foreign_attribute(
        df,
        foreign_attribute_name=foreign_attribute_name,
        local_attribute_name=local_attribute_name,
        foreign_attribute_values=foreign_attribute_values,
        foreign_attribute_to_local_attribute_relation=foreign_attribute_to_local_attribute_relation,
    )

    assert sorted(filtered[pk_col].values.tolist()) == sorted(expected_pk_col_values)
