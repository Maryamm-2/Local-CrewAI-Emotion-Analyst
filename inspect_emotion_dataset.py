# inspect_emotion_dataset.py
"""
Dataset Inspector Utility
-------------------------
A standalone utility script to explore the properties of the Hugging Face 'emotion' dataset.
Useful for developers to understand the data schema before building agents.

Features:
- Prints cache location.
- Lists dataset features (columns, types).
- Displays sample rows.
- Shows label mappings (int -> string).
"""

from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("emotion", split="train")

# Print cache location
cache_files = dataset.cache_files
print("Dataset cache location:")
for f in cache_files:
    print(f["filename"])

# Print dataset features
print("\nDataset features:")
print(dataset.features)

# Convert to DataFrame and print sample rows
print("\nSample rows:")
df = pd.DataFrame(dataset)
print(df.head())

# Print dataset shape and size info
print("\nDataset shape and size:")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print(f"Column names: {list(df.columns)}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

# Print label mapping
if "label" in dataset.features:
    print("\nLabel mapping:")
    for idx, name in enumerate(dataset.features["label"].names):
        print(f"{idx}: {name}")
