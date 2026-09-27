"""
Data Discovery (Recursively) and ingests the data parallelly.
Author: Albert Jojo
"""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv

_QueueItem = tuple[int, str, pd.DataFrame | BaseException | None]


# def retrieve_files(file_paths: list[str]) -> pd.DataFrame:
#     """Read and concatenates a list of csv paths"""
#     if not file_paths:
#         raise ValueError("file_paths cannot be empty")
#     with ThreadPoolExecutor(max_workers=min(4, len(file_paths))) as pool:
#         frames = list(pool.map(pd.read_csv, file_paths))
#     return pd.concat(frames, ignore_index=True)


@dataclass
class DataIngestor:
    root_dir: Path
    prefixes: tuple[str, ...] = ()
    max_workers: int = 4
    chunksize: int = 250_000
    block_size: int = 64 * 1024 * 1024  # bytes per batch
    queue_size: int = 8
    validate_schema: bool = True

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        if self.max_workers < 1:
            raise ValueError("max_workers >= 1")
        if self.block_size < 1:
            raise ValueError("chunk_size >= 1")
        if self.queue_size < 1:
            raise ValueError("queue_size must be atleast 1.")

    def read_schema(self, path: Path) -> pa.Schema:
        reader = pv.open_csv(
            path, read_options=pv.ReadOptions(block_size=self.block_size)
        )
        return reader.schema

    def discover_files(self) -> list[Path]:
        """Recursively find CSV files matching the configured prefixes"""
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.root_dir}")

        files = sorted(
            path
            for path in self.root_dir.rglob("*.csv")
            if not self.prefixes or path.name.startswith(self.prefixes)
        )
        if not files:
            raise FileNotFoundError(
                f"No matching CSV files found under {self.root_dir}"
            )
        return files

    def _read_file(
        self, file_index: int, path: Path, output: Queue[_QueueItem]
    ) -> None:
        """Read one file lazily and put its chunks into the bounded queue"""
        try:
            reader = pv.open_csv(
                path,
                read_options=pv.ReadOptions(block_size=self.block_size),
            )
            for batch in reader:
                output.put(
                    (file_index, "chunk", pa.Table.from_batches([batch]).to_pandas())
                )
        except BaseException as error:
            output.put((file_index, "error", error))
        finally:
            output.put((file_index, "done", None))

    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        """Yields chunks as workers finish reading them, in arrival order (not file order)"""
        files = self.discover_files()
        output: Queue[_QueueItem] = Queue(maxsize=self.queue_size)
        worker_count = min(self.max_workers, len(files))
        remaining = len(files)
        errors: list[tuple[Path, BaseException]] = []

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            for file_index, path in enumerate(files):
                pool.submit(self._read_file, file_index, path, output)

            while remaining > 0:
                match output.get():
                    case (_, "chunk", pd.DataFrame() as chunk):
                        yield chunk
                    case (file_index, "error", BaseException() as error):
                        errors.append((files[file_index], error))
                    case (_, "done", _):
                        remaining -= 1
                    case item:
                        raise TypeError(f"Unexpected queue item: {item!r}")

        if errors:
            path, error = errors[0]
            raise RuntimeError(f"Failed to read CSV file: {path}") from error

    def load(self) -> pd.DataFrame:
        """Materialize all chunks into one dataframe"""
        return pd.concat(self.iter_chunks(), ignore_index=True)
