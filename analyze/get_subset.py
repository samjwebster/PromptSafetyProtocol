import os
import json
import random

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

data_dir = "/afs/crc.nd.edu/user/s/swebster/Private/SecureSoftwareEngineering/compose_json"

datasets = {
    "tensortrust": os.path.join(data_dir, "tensortrust_full.json"),
    "tt_extraction": os.path.join(data_dir, "tt_extraction_full.json"),
    "train_deepset": os.path.join(data_dir, "train_deepset_full.json"),
    "test_deepset": os.path.join(data_dir, "test_deepset_full.json"),
}

responses_dir = "/afs/crc.nd.edu/user/s/swebster/Private/SecureSoftwareEngineering/security_protocol"
# concise_datasets = dict()
# for key in datasets.keys():
#     concise_datasets[key] = load_json(os.path.join(responses_dir, key + "_concise_12-3.json"))

# verbose_datasets = dict()
# for key in datasets.keys():
#     verbose_datasets[key] = load_json(os.path.join(responses_dir, key + "_verbose_12-3.json"))

baseline_datasets = dict()
for key in datasets.keys():
    baseline_datasets[key] = load_json(os.path.join(responses_dir, key + "_baseline.json"))

# Number of samples to select
selection_counts = {
    "tensortrust": 25,
    "tt_extraction": 25,
    "train_deepset": {"injection": 13, "safe": 13},
    "test_deepset": {"injection": 12, "safe": 12},
}

# Load all datasets
loaded_datasets = {name: load_json(path) for name, path in datasets.items()}

selected_prompts = {}

# Select samples from tensortrust (all hijacking)
# tensortrust_ids = [item["id"] for item in loaded_datasets["tensortrust"]]
tensortrust_ids = list(loaded_datasets["tensortrust"].keys())
selected_prompts["tensortrust"] = random.sample(tensortrust_ids, selection_counts["tensortrust"])

# Select samples from tt_extraction (all extraction)
# tt_extraction_ids = [item["id"] for item in loaded_datasets["tt_extraction"]]
tt_extraction_ids = list(loaded_datasets["tt_extraction"].keys())
selected_prompts["tt_extraction"] = random.sample(tt_extraction_ids, selection_counts["tt_extraction"])

# Select samples from train_deepset and test_deepset
for dataset_name in ["train_deepset", "test_deepset"]:
    data = loaded_datasets[dataset_name]
    
    # Separate injection and safe samples
    # injection_samples = [item["id"] for item in data if item["label"] == 1]
    # safe_samples = [item["id"] for item in data if item["label"] == 0]

    injection_samples = [id for id,entry in data.items() if entry["label"] == 1]
    safe_samples = [id for id,entry in data.items() if entry["label"] == 0]
    
    # Sample from both categories
    selected_injection = random.sample(
        injection_samples, selection_counts[dataset_name]["injection"]
    )
    selected_safe = random.sample(safe_samples, selection_counts[dataset_name]["safe"])

    # Combine selections
    selected_prompts[dataset_name] = selected_injection + selected_safe

# Get prompts and responses for each config
concise_subset = dict()
verbose_subset = dict()
baseline_subset = dict()
print(loaded_datasets.keys())
for key in datasets.keys():
    ids = selected_prompts[key]

    # concise_subset[key] = [{"prompt": loaded_datasets[key][id]["prompt"], "response": concise_datasets[key][id], "id": id, "label": loaded_datasets[key][id]["label"], "accept": None, "safety_mentioned": None} for id in ids]
    # verbose_subset[key] = [{"prompt": loaded_datasets[key][id]["prompt"], "response": verbose_datasets[key][id], "id": id, "label": loaded_datasets[key][id]["label"], "accept": None, "safety_mentioned": None} for id in ids]

    baseline_subset[key] = [{"prompt": loaded_datasets[key][id]["prompt"], "response": baseline_datasets[key][id], "id": id, "label": loaded_datasets[key][id]["label"], "accept": None, "safety_mentioned": None} for id in ids]

# outpath = "./subsets/concise.json"
# save_json(outpath, concise_subset)

# outpath = "./subsets/verbose.json"
# save_json(outpath, verbose_subset)

# outpath = "./subsets/baseline.json"
# save_json(outpath, baseline_subset)
