import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.preprocessing.data_ingestion import DataIngestor
from src.preprocessing.ingest_data import save_dataset


class TestDataIngestion(unittest.TestCase):
    def write_source_files(self, root_dir: Path) -> None:
        (root_dir / "source_one.csv").write_text(
            "STUDENT_ID,ITEM_TYPE,value\n001,00001234,one\n002,00005678,two\n",
            encoding="utf-8",
        )
        (root_dir / "source_two.csv").write_text(
            "STUDENT_ID,ITEM_TYPE,value\n003,00009876,three\n",
            encoding="utf-8",
        )

    def test_ingestion_preserves_ids_and_codes_and_writes_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root_dir = Path(temporary_directory)
            self.write_source_files(root_dir)
            ingestor = DataIngestor(root_dir=root_dir, prefixes=("source_",))

            self.assertEqual(
                ingestor.count_rows_by_file(),
                {
                    root_dir / "source_one.csv": 2,
                    root_dir / "source_two.csv": 1,
                },
            )
            loaded = ingestor.load()
            self.assertEqual(loaded["STUDENT_ID"].tolist(), ["001", "002", "003"])
            self.assertEqual(
                loaded["ITEM_TYPE"].tolist(), ["00001234", "00005678", "00009876"]
            )

            output_path = root_dir / "combined.parquet"
            save_dataset(root_dir, ("source_",), output_path)
            restored = pd.read_parquet(output_path)
            self.assertEqual(len(restored), 3)
            self.assertEqual(
                sorted(restored["ITEM_TYPE"].tolist()),
                sorted(loaded["ITEM_TYPE"].tolist()),
            )

    def test_row_count_mismatch_removes_incomplete_combined_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root_dir = Path(temporary_directory)
            self.write_source_files(root_dir)
            output_path = root_dir / "combined.parquet"

            with (
                patch.object(
                    DataIngestor,
                    "iter_chunks",
                    return_value=iter(
                        [pd.DataFrame({"STUDENT_ID": ["001"], "value": ["one"]})]
                    ),
                ),
                self.assertRaisesRegex(AssertionError, "Row count mismatch"),
            ):
                save_dataset(root_dir, ("source_",), output_path)

            self.assertFalse(output_path.exists())
