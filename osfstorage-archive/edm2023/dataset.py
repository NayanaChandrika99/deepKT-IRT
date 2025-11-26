from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Set

import pandas as pd
from pydantic import BaseModel


class Dataset(BaseModel):
    """Dataset"""

    class Config:
        """Config"""

        arbitrary_types_allowed = True

    @classmethod
    def from_dir(cls, dir_path: Path) -> "Dataset":
        """Load dataset from directory"""
        data = {}
        for fpath in dir_path.iterdir():
            if fpath.suffix == ".csv" and str(fpath.stem) in cls.__fields__.keys():
                data[str(fpath.stem)] = pd.read_csv(fpath)

        return cls(**data)


class RawData(Dataset):
    """Schema for Raw Data"""

    evaluation_unit_test_scores: pd.DataFrame
    assignment_relationships: pd.DataFrame
    training_unit_test_scores: pd.DataFrame


class ProcessedData(Dataset):
    """Schema for processed data"""

    edm_cup_2023_in_unit_assignment_scorable_actions: pd.DataFrame
    edm_cup_2023_in_unit_problem_item_response_elasped_time: pd.DataFrame


class IRelation(ABC):
    """Relation establish the relation between attributes"""

    @abstractmethod
    def source_attribute_name(self) -> str:
        """Get source attribute name"""

    @abstractmethod
    def target_attribute_name(self) -> str:
        """Get target attribute name"""

    @abstractmethod
    def get_target_attribute_values(self, source_attr_val: Any) -> Set[Any]:
        """Get target attribute value based on source attribute value"""


class DataFrameFilter:
    """Filter data set by user id"""

    @staticmethod
    def filter_by_local_attribute(
        df: pd.DataFrame, local_attribute_name: str, local_attribute_vals: Set[Any]
    ) -> pd.DataFrame:
        """Filter df by local attribute"""

        mask = df[local_attribute_name].isin(local_attribute_vals)

        return df[mask].copy()

    @staticmethod
    def filter_by_foreign_attribute(
        df: pd.DataFrame,
        foreign_attribute_name: str,
        local_attribute_name: str,
        foreign_attribute_values: Set[Any],
        foreign_attribute_to_local_attribute_relation: IRelation,
    ):
        """Filter df by foreign attribute"""
        pool = set([])

        for foreign_attribute_value in foreign_attribute_values:
            local_attribute_values = foreign_attribute_to_local_attribute_relation.get_target_attribute_values(
                foreign_attribute_value
            )
            pool = pool.union(local_attribute_values)

        return DataFrameFilter.filter_by_local_attribute(df, local_attribute_name, pool)


class TestLogIdToAssignmentLogIdRelation(IRelation):
    """Relation between `unit_test_assignment_log_id` and `in_unit_assignment_log_id`"""

    def __init__(self, assignment_relationships: pd.DataFrame):
        # `in_unit_assignment_log_id` as named as `assignment_log_id` in every dataframe except assignment_relationships
        agged = assignment_relationships.groupby(self.source_attribute_name)[
            "in_unit_assignment_log_id"
        ].apply(lambda x: set(x.tolist()))
        self.source_to_target_dict = agged.to_dict()

    @property
    def source_attribute_name(self) -> str:
        """Source attribute name"""
        return "unit_test_assignment_log_id"

    @property
    def target_attribute_name(self) -> str:
        """Target attribute name"""
        return "assignment_log_id"

    def get_target_attribute_values(self, source_attr_val: Any) -> Set[Any]:
        """Get target attribute values based on source attribute value"""
        if source_attr_val in self.source_to_target_dict:
            return self.source_to_target_dict[source_attr_val]
        else:
            return set([])
