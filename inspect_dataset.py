import pyarrow.parquet as pq
from pathlib import Path

# Locate dataset
files = list(Path("dataset").rglob("traces.parquet"))

if not files:
    raise FileNotFoundError("traces.parquet not found")

file_path = files[0]

# Read Parquet metadata without loading all records
parquet_file = pq.ParquetFile(file_path)

print("Dataset:", file_path)
print("Total rows:", parquet_file.metadata.num_rows)
print("Total row groups:", parquet_file.num_row_groups)

print("\nColumns:")
print(parquet_file.schema.names)

print("\nSchema:")
print(parquet_file.schema)

# Read only 5 records and selected columns
selected_columns = [
    "trace_id",
    "span_id",
    "parent_span_id",
    "call_index",
    "session_call_count",
    "model",
    "input_tokens",
    "output_tokens",
    "usage_basis"
]

print("\nFirst 5 records:")

batches = parquet_file.iter_batches(
    batch_size=5,
    columns=selected_columns
)

first_batch = next(batches)
print(first_batch.to_pandas().to_string(index=False))
