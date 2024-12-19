import pandas as pd

print("Loading attacks")

attacks_dataset = pd.read_json(
    "https://raw.githubusercontent.com/HumanCompatibleAI/tensor-trust-data/main/benchmarks/extraction-robustness/v1/extraction_robustness_dataset.jsonl",
    lines=True,
)

print("Attacks loaded. Filtering columns")

# Select only "sample_id" and "attack" columns
filtered_dataset = attacks_dataset[["sample_id", "attack"]].rename(columns={"sample_id": "id", "attack": "prompt"})

print(f"Found {len(filtered_dataset)} attacks. Filtering null attacks")

filtered_dataset = filtered_dataset.dropna(subset=["prompt"])

print(f"Found {len(filtered_dataset)} non-null attacks")

print("Dropping duplicate attacks")


filtered_dataset = filtered_dataset.drop_duplicates(["prompt"])


print("Saving to new json")

# Save to a new JSON file
filtered_dataset.to_json("tensortrust.json", orient="records")