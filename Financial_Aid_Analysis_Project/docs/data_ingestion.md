# Data Ingestion


### Overview
This document contains the thought process, architecture behind my data ingestion class plugin.

When the ingestion script writes a combined CSV, it counts rows in each matching
source file and asserts that their sum matches the row count in the combined
file. If the counts differ, the script raises an assertion error and removes the
incomplete combined output.

