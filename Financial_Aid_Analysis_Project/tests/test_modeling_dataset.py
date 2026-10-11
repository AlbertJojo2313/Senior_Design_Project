import json
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
                "RETAINED_YR2": [index // 2 % 2 for index in range(row_count)],
                "Graduated Count Yr 3": [
                    index // 2 % 2 for index in range(row_count)
                ],
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

    def test_creates_views_with_disjoint_reproducible_split_and_cip2(self) -> None:
        master = self.make_master()

        datasets = self.modeling.create_datasets({"master": master})
        split = pd.read_parquet(self.project_dir / "data/derived/split_v1.parquet")
        train_ids = set(split.loc[split["split"].eq("train"), "STUDENT_ID"])
        test_ids = set(split.loc[split["split"].eq("test"), "STUDENT_ID"])

        self.assertFalse(train_ids & test_ids)
        self.assertEqual(train_ids | test_ids, set(master["STUDENT_ID"]))
        self.assertEqual(set(datasets), set(self.modeling.cfg["views"]))
        self.assertIn("CIP2", datasets["ftic_ret1"]["X_train"].columns)
        self.assertEqual(datasets["ftic_ret1"]["X_train"]["CIP2"].iloc[0], "11")
        self.assertEqual(
            datasets["ftic_ret1"]["X_train_transformed"].shape[1],
            datasets["ftic_ret1"]["X_test_transformed"].shape[1],
        )
        self.assertTrue(
            (self.project_dir / "data/derived/view_ftic_ret1_train.parquet").is_file()
        )
        self.assertTrue(
            (self.project_dir / "data/derived/view_ftic_ret1_test.parquet").is_file()
        )

        with (self.project_dir / "data/derived/views_config.json").open() as config_file:
            views_config = json.load(config_file)
        self.assertEqual(
            views_config["ftic_grad3"]["outcome"], "Graduated Count Yr 3"
        )
        self.assertEqual(
            views_config["ftic_ret1"]["treatment"][0], "aid_yr1_src_federal"
        )

        previous_split = split.copy()
        self.modeling.create_datasets({"master": master})
        repeated_split = pd.read_parquet(
            self.project_dir / "data/derived/split_v1.parquet"
        )
        pd.testing.assert_frame_equal(previous_split, repeated_split)

    def test_reports_missing_outcome_column(self) -> None:
        master = self.make_master().drop(columns="RETAINED_YR2")

        with self.assertRaisesRegex(KeyError, "RETAINED_YR2"):
            self.modeling.create_datasets({"master": master})

    def test_rejects_duplicate_student_ids(self) -> None:
        master = self.make_master()
        master.loc[1, "STUDENT_ID"] = master.loc[0, "STUDENT_ID"]

        with self.assertRaisesRegex(ValueError, "STUDENT_ID must be unique"):
            self.modeling.create_datasets({"master": master})
