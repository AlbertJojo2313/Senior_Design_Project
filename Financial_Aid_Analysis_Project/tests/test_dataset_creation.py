import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.preprocessing.dataset_creation import DatasetCreation
from src.preprocessing.eda_dataset_creation import EDADatasetCreation
from src.preprocessing.ml_dataset_creation import MLDatasetCreation


def write_config(project_dir: Path, filename: str, content: str) -> Path:
    config_dir = project_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / filename
    config_path.write_text(content, encoding="utf-8")
    return config_path


class DatasetCreationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_dir = Path(self.temporary_directory.name)
        self.ml_config = write_config(
            self.project_dir,
            "ml.yaml",
            """\
expected_rows: null
aid_years: [1, 4]
term_years: [1, 4]
aid_sources: [federal, institutional, private, state, other]
term_drop_cols: []
first_cols:
  - STUDENT_ID
  - COHORT_YEAR
  - CF_BOE_TERM_ID
  - TERM_DESCR
  - ACAD_YEAR
  - CF_BOE_NM_IND
  - Credits_Transfered
grad_flag_cols:
  - Graduated Count Yr 3
  - Graduated Count Yr 4
  - Graduated Count Yr 5
  - Graduated Count Yr 6
sat_zero_is_missing: true
post_merge_drop_cols:
  - BOE_RECENT_TYPE
  - BOE_RECENT_TYPE_DESCR
  - BOE_RECENT_TYPE_DATE
output_columns: [STUDENT_ID, Entry_CIP, AA_AS, SPRING_FULL_TIME, SPRING_PART_TIME, SAT_TOTAL, SAT_VERB, SAT_MATH, re_entered, UCF_First_Term_GPA]
output_path: data/derived/ml_dataset.parquet
""",
        )
        self.ml = MLDatasetCreation(
            config_path=self.ml_config, project_dir=self.project_dir
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()


class TestDatasetSourceLoading(DatasetCreationTestCase):
    def test_base_class_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            DatasetCreation()  # type: ignore[abstract]

    def test_load_sources_normalizes_keys_and_filters_recipients(self) -> None:
        raw_dir = self.project_dir / "data" / "raw_data"
        combined_dir = self.project_dir / "data" / "combined"
        raw_dir.mkdir(parents=True)
        combined_dir.mkdir(parents=True)
        (combined_dir / "financial_aid.csv").write_text(
            "STUDENT_ID,ITEM_TYPE,Aid_Year\n 001 ,00001234,2020\n",
            encoding="utf-8",
        )
        (combined_dir / "student_term.csv").write_text(
            "STUDENT_ID,CF_BOE_TERM_ID\n001,202001\n999,202001\n",
            encoding="utf-8",
        )
        (raw_dir / "Student_Static_Data.csv").write_text(
            "STUDENT_ID,COHORT_YEAR,STUDENT_TYPE\n001,2019-2020,A\n999,2019-2020,B\n",
            encoding="utf-8",
        )

        sources = self.ml.load_sources()
        self.assertEqual(sources["financial_aid"]["ITEM_TYPE"].iloc[0], "00001234")
        self.assertEqual(sources["student_term"]["STUDENT_ID"].tolist(), ["001"])
        self.assertEqual(sources["student_static"]["STUDENT_ID"].tolist(), ["001"])
        self.assertEqual(sources["financial_aid"]["AID_YEAR"].iloc[0], 2020)


class TestMLDatasetCreation(DatasetCreationTestCase):
    def test_aid_wide_has_fixed_columns_and_year_placement(self) -> None:
        static = pd.DataFrame(
            {
                "STUDENT_ID": ["s1", "s2"],
                "COHORT_YEAR": ["2020-2021", "2020-2021"],
            }
        )
        aid = pd.DataFrame(
            {
                "STUDENT_ID": ["s1", "s2", "s1"],
                "COHORT_YEAR": ["2020-2021", "2020-2021", "2020-2021"],
                "AID_YEAR": [2021, 2022, 2021],
                "FA_SOURCE_DESCR": ["Federal", "State", "Other"],
            }
        )

        wide = self.ml.build_aid_wide(aid, static).set_index("STUDENT_ID")
        self.assertEqual(wide.loc["s2", "aid_yr1_src_federal"], 0)
        self.assertEqual(wide.loc["s2", "aid_yr2_src_state"], 1)
        self.assertEqual(wide.loc["s1", "aid_yr1_src_other"], 1)
        self.assertEqual(wide.loc["s1", "aid_yr1_src_private"], 0)
        self.assertEqual(len(wide.columns), 20)
        self.assertTrue(wide.index.is_unique)

    def test_prepare_term_keeps_most_complete_duplicate(self) -> None:
        term = pd.DataFrame(
            {
                "STUDENT_ID": ["s1", "s1"],
                "CF_BOE_TERM_ID": ["202001", "202001"],
                "ACAD_YEAR": ["2019-20", "2019-20"],
                "value": [None, "complete"],
            }
        )

        prepared = self.ml.prepare_term(term)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared["value"].iloc[0], "complete")

    def test_first_term_does_not_fill_a_null_from_a_later_term(self) -> None:
        term = pd.DataFrame(
            {
                "STUDENT_ID": ["s1", "s1"],
                "COHORT_YEAR": ["2019-2020", "2019-2020"],
                "CF_BOE_TERM_ID": ["201908", "202001"],
                "TERM_DESCR": ["Fall", "Spring"],
                "ACAD_YEAR": ["2019-20", "2019-20"],
                "CF_BOE_NM_IND": [None, "later"],
                "Credits_Transfered": [3, 4],
            }
        )

        first = self.ml.build_first_term(term)
        self.assertTrue(pd.isna(first["CF_BOE_NM_IND"].iloc[0]))
        self.assertEqual(first["Credits_Transfered_Max"].iloc[0], 4)

    def test_term_year_mapping_uses_static_cohort(self) -> None:
        term = pd.DataFrame(
            {
                "STUDENT_ID": ["s1", "s1"],
                "ACAD_YEAR": ["2019-20", "2020-21"],
            }
        )
        static = pd.DataFrame({"STUDENT_ID": ["s1"], "COHORT_YEAR": ["2019-2020"]})

        with_year = self.ml.add_term_year(term, static)
        self.assertEqual(with_year["term_yr"].tolist(), [1, 2])

    def test_year_summaries_keep_absent_year_metrics_missing(self) -> None:
        term = pd.DataFrame(
            {
                "STUDENT_ID": ["s1"],
                "CF_BOE_TERM_ID": ["201908"],
                "term_yr": [1],
                "CUM_TOTHRS_EARN_BEGIN_TERM": [0.0],
                "Credits_Earned_In_Term": [12.0],
                "CF_BOE_TERM_CRS_LD": [12.0],
                "CUM_GPA_END_TERM": [3.5],
                "UCF_GPA_END_TERM": [3.5],
            }
        )

        summary = self.ml.build_year_summaries(term).iloc[0]
        self.assertTrue(pd.isna(summary["yr3_n_terms"]))
        self.assertTrue(pd.isna(summary["yr3_end_cum_gpa"]))
        self.assertEqual(summary["yr1_n_terms"], 1)

    def test_master_keeps_one_row_per_recipient_and_fills_term_counts(self) -> None:
        static = pd.DataFrame(
            {
                "STUDENT_ID": ["s1", "s2"],
                "COHORT_YEAR": ["2019-2020", "2019-2020"],
            }
        )
        aid_wide = pd.DataFrame(
            {
                "STUDENT_ID": ["s1", "s2"],
                "aid_yr1_src_federal": pd.Series([1, 0], dtype="Int8"),
            }
        )
        first_term = pd.DataFrame(
            {
                "STUDENT_ID": ["s1", "s2"],
                "COHORT_YEAR_term": ["2019-2020", "2019-2020"],
                "first_term_id": ["201908", "201908"],
            }
        )
        by_year = pd.DataFrame(
            {
                "STUDENT_ID": ["s1"],
                "yr1_n_terms": [2],
                "yr3_end_cum_gpa": [None],
            }
        )
        self.ml.aid_mismatch_ids = set()

        master = self.ml.build_master(static, aid_wide, first_term, by_year)
        self.assertEqual(len(master), 2)
        self.assertTrue(master["STUDENT_ID"].is_unique)
        self.assertTrue(master["first_term_id"].notna().all())
        self.assertEqual(master.loc[1, "yr1_n_terms"], 0)
        self.assertTrue(pd.isna(master.loc[1, "yr3_end_cum_gpa"]))

    def test_post_cleanup_preserves_nulls_except_sat_zero_recoding(self) -> None:
        row = {
            "STUDENT_ID": "s1",
            "COHORT_YEAR": "2019-2020",
            "Entry_Age": 18,
            "Entry_Date": "201908",
            "STUDENT_TYPE": "A",
            "STUDENT_TYPE_DESCR": "Transfer",
            "Entry_HS_GPA": 4.0,
            "AA_AS": 1,
            "Entry_CIP": "1234",
            "RETAINED_YR1": 1,
            "RETAINED_YR2": 1,
            "RETAINED_YR3": 0,
            "DEGREE_TERM": None,
            "DEG_OUT_YR": None,
            "Graduated Count Yr 3": 0,
            "Graduated Count Yr 4": 0,
            "Graduated Count Yr 5": 0,
            "Graduated Count Yr 6": 0,
            "SUMMER_FULL_TIME": 1,
            "SUMMER_PART_TIME": 0,
            "FALL_FULL_TIME": 1,
            "FALL_PART_TIME": 0,
            "SPRING_FULL_TIME": 1,
            "SPRING_PART_TIME": 0,
            "SAT_TOTAL": 0,
            "SAT_VERB": 0,
            "SAT_MATH": 0,
            "ACT_TOTAL": None,
            "ACT_ENG": None,
            "ACT_MATH": None,
            "ACT_READ": None,
            "Transfer_GPA": None,
            "UCF_First_Term_GPA": None,
            "Transfer_Shock": None,
            "BOE_RECENT_TYPE": "x",
            "BOE_RECENT_TYPE_DESCR": "x",
            "BOE_RECENT_TYPE_DATE": "201908",
            "CF_BOE_NM_IND": 1,
            "cohort_mismatch_term": False,
            "cohort_mismatch_aid": False,
        }
        source = pd.DataFrame([row])
        cleaned = self.ml.post_merge_cleanup(source)

        self.assertTrue(
            cleaned[["SAT_TOTAL", "SAT_VERB", "SAT_MATH"]].isna().all().all()
        )
        self.assertEqual(cleaned["Entry_CIP"].iloc[0], "001234")
        self.assertEqual(str(cleaned["AA_AS"].dtype), "Int8")
        self.assertFalse(cleaned["re_entered"].iloc[0])
        expected_nulls = source.reindex(columns=cleaned.columns).isna().sum()
        expected_nulls["re_entered"] = 0
        expected_nulls[["SAT_TOTAL", "SAT_VERB", "SAT_MATH"]] += 1
        self.assertEqual(
            cleaned.isna().sum().to_dict(),
            expected_nulls.to_dict(),
        )
        self.assertEqual(cleaned.columns.tolist(), self.ml.cfg["output_columns"])
        self.assertEqual(cleaned["STUDENT_ID"].iloc[0], "s1")
        self.assertEqual(cleaned["Entry_CIP"].iloc[0], "001234")
        self.assertEqual(cleaned["AA_AS"].iloc[0], 1)
        self.assertEqual(cleaned["SPRING_FULL_TIME"].iloc[0], 1)
        self.assertEqual(cleaned["SPRING_PART_TIME"].iloc[0], 0)
        self.assertEqual(str(cleaned["SPRING_FULL_TIME"].dtype), "Int8")
        self.assertTrue(pd.isna(cleaned["SAT_VERB"].iloc[0]))
        self.assertTrue(
            {"BOE_RECENT_TYPE", "BOE_RECENT_TYPE_DATE"}.isdisjoint(cleaned.columns)
        )


class TestEDADatasetCreation(DatasetCreationTestCase):
    def test_eda_splits_cover_unsplit_rows_and_drop_merge_indicator(self) -> None:
        config = write_config(
            self.project_dir,
            "eda.yaml",
            "output_dir: data/derived/eda\n",
        )
        eda = EDADatasetCreation(config_path=config, project_dir=self.project_dir)
        static = pd.DataFrame(
            {
                "STUDENT_ID": ["s1", "s2"],
                "STUDENT_TYPE": ["A", "B"],
                "static_value": [10, 20],
            }
        )
        sources = {
            "student_static": static,
            "student_term": pd.DataFrame(
                {
                    "STUDENT_ID": ["s1", "s2", "s2"],
                    "term_value": [1, 2, 3],
                }
            ),
            "financial_aid": pd.DataFrame(
                {
                    "STUDENT_ID": ["s1", "s1", "s2"],
                    "aid_value": [4, 5, 6],
                }
            ),
        }

        outputs = eda.create_datasets(sources)
        self.assertEqual(len(outputs["eda_term_A"]) + len(outputs["eda_term_B"]), 3)
        self.assertEqual(len(outputs["eda_aid_A"]) + len(outputs["eda_aid_B"]), 3)
        for output in outputs.values():
            self.assertNotIn("_merge", output.columns)
            self.assertLessEqual(output["STUDENT_TYPE"].nunique(), 1)
        self.assertTrue(eda.output_path("eda_term_A").is_file())
