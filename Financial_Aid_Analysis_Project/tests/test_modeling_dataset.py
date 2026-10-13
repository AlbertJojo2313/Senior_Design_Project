import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.preprocessing.modeling_dataset import ModelingDatasetCreation


class TestModelingDatasetCreation(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_dir = Path(self.temporary_directory.name)
        self.modeling = ModelingDatasetCreation(
            config_path=Path(__file__).parents[1] / "configs" / "model_views.yaml",
            project_dir=self.project_dir,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_master(self) -> pd.DataFrame:
        row_count = 80
        master = pd.DataFrame(
            {
                "STUDENT_ID": [f"s{index:03}" for index in range(row_count)],
                "STUDENT_TYPE": ["A", "B"] * (row_count // 2),
                "RETAINED_YR1": [index // 2 % 2 for index in range(row_count)],
                "RETAINED_YR2": [index // 4 % 2 for index in range(row_count)],
                "RETAINED_YR3": [index // 8 % 2 for index in range(row_count)],
                "Graduated Count Yr 3": [index // 2 % 2 for index in range(row_count)],
                "Graduated Count Yr 4": [
                    1 - index // 2 % 2 for index in range(row_count)
                ],
                "COHORT_YEAR": ["2020-2021"] * row_count,
                "Entry_CIP": ["110101"] * row_count,
                "Entry_Age": [18 + index % 4 for index in range(row_count)],
                "Entry_HS_GPA": [3.0 + (index % 10) / 10 for index in range(row_count)],
                "ACT_TOTAL": [20 + index % 10 for index in range(row_count)],
                "Credits_Transfered_Max": [index % 15 for index in range(row_count)],
            }
        )
        for year in range(1, 5):
            for source in self.modeling.cfg["aid_sources"]:
                master[f"aid_yr{year}_src_{source}"] = 1
        return master

    def test_ftic_ret3_requires_both_prior_retention_outcomes(self) -> None:
        master = self.make_master()
        datasets = self.modeling.create_datasets({"master": master})
        ret3 = datasets["ftic_ret3"]
        sample_count = len(ret3["y_train"]) + len(ret3["y_test"])

        self.assertEqual(sample_count, 10)
        self.assertEqual(int(ret3["y_train"].sum() + ret3["y_test"].sum()), 5)

    def test_nullable_features_are_normalized_before_imputation(self) -> None:
        master = self.make_master()
        master["Entry_Age"] = pd.Series(master["Entry_Age"], dtype="Int64")
        master.loc[[1, 3, 5, 7], "Entry_Age"] = pd.NA
        master["Entry_HS_GPA"] = pd.Series(master["Entry_HS_GPA"], dtype="Float64")
        master.loc[[1, 3, 5, 7], "Entry_HS_GPA"] = pd.NA
        master["COHORT_YEAR"] = pd.Series(master["COHORT_YEAR"], dtype="string")
        master.loc[[1, 3, 5, 7], "COHORT_YEAR"] = pd.NA
        parquet_path = self.project_dir / "nullable_master.parquet"
        master.to_parquet(parquet_path, index=False)
        master = pd.read_parquet(parquet_path)

        datasets = self.modeling.create_datasets({"master": master})
        ftic = datasets["ftic_ret1"]

        self.assertEqual(ftic["X_train"]["Entry_Age"].dtype, "float64")
        self.assertEqual(ftic["X_train"]["COHORT_YEAR"].dtype, "object")
        self.assertTrue(ftic["X_train_transformed"].size > 0)

        raw_features = ftic["X_train"].copy()
        raw_features["Entry_Age"] = raw_features["Entry_Age"].astype("Int64")
        raw_features.loc[raw_features.index[0], "Entry_Age"] = pd.NA
        raw_features["Entry_HS_GPA"] = raw_features["Entry_HS_GPA"].astype("Float64")
        raw_features.loc[raw_features.index[0], "Entry_HS_GPA"] = pd.NA
        raw_features["COHORT_YEAR"] = raw_features["COHORT_YEAR"].astype("string")
        raw_features.loc[raw_features.index[0], "COHORT_YEAR"] = pd.NA
        cfg = self.modeling.view_configs()["ftic_ret1"]
        transformed = self.modeling.make_preprocessor(cfg).fit_transform(raw_features)
        self.assertTrue(transformed.size > 0)
