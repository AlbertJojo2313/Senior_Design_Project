"""
   This script contains the abstract class for dataset creation for eda and model.
   Author: Albert

Loads the configuration file for each class yaml and the combined parquet or CSV files.
Normalizes the student id, and has a save function for both the classes to either parquet or csv.
Also does verification
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.preprocessing.data_ingestion import DataIngestor


class DatasetCreation(ABC):
    """Load and align the source tables shared by EDA and ML datasets."""

    config_name: str

    def __init__(
        self,
        config_path: str | Path | None = None,
        project_dir: str | Path | None = None,
    ) -> None:
        self.project_dir = (
            Path(project_dir).resolve()
            if project_dir is not None
            else Path(__file__).resolve().parents[2]
        )
        self.config_path = (
            Path(config_path)
            if config_path is not None
            else self.project_dir / "configs" / self.config_name
        )
        with self.config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if not isinstance(config, dict):
            raise TypeError(f"Expected a YAML mapping in {self.config_path}")
        self.cfg: dict[str, Any] = config
        self.logger = logging.getLogger(type(self).__module__)

    def log(self, message: str) -> None:
        self.logger.info(message)

    @abstractmethod
    def create_datasets(self, sources: dict[str, pd.DataFrame] | None = None) -> Any:
        """Build and persist this dataset class's outputs."""

    @staticmethod
    def _read_table(path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path, dtype={"STUDENT_ID": str, "ITEM_TYPE": str})

    def _load_combined(
        self, dataset_name: str, source_prefix: tuple[str, ...]
    ) -> pd.DataFrame:
        combined_dir = self.project_dir / "data" / "combined"
        for suffix in (".parquet", ".csv"):
            path = combined_dir / f"{dataset_name}{suffix}"
            if path.is_file():
                return self._read_table(path)
        raw_dir = self.project_dir / "data" / "raw_data"
        return DataIngestor(root_dir=raw_dir, prefixes=source_prefix).load()

    def load_sources(self) -> dict[str, pd.DataFrame]:
        """Load sources, preserve key/code text, and restrict to aid recipients."""
        aid = self._load_combined("financial_aid", ("FINAID_",))
        term = self._load_combined("student_term", ("STUDENT_TERM_",))
        static_path = self.project_dir / "data" / "raw_data" / "Student_Static_Data.csv"
        static = self._read_table(static_path)

        if "Aid_Year" in aid and "AID_YEAR" not in aid:
            aid = aid.rename(columns={"Aid_Year": "AID_YEAR"})

        tables = {"financial_aid": aid, "student_term": term, "student_static": static}
        for table_name, frame in tables.items():
            if "STUDENT_ID" not in frame:
                raise KeyError(f"{table_name} is missing required STUDENT_ID column")
            frame["STUDENT_ID"] = frame["STUDENT_ID"].astype(str).str.strip()
        if not tables["student_static"]["STUDENT_ID"].is_unique:
            raise ValueError("student_static must have one row per STUDENT_ID")

        recipients = set(aid["STUDENT_ID"].dropna())
        for table_name in ("student_term", "student_static"):
            frame = tables[table_name]
            before_ids = set(frame["STUDENT_ID"].dropna())
            tables[table_name] = frame[frame["STUDENT_ID"].isin(recipients)].copy()
            removed = len(before_ids - recipients)
            self.log(f"{table_name}: removed {removed} non-recipient student IDs")

        return tables
