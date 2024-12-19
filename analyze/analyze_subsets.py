import os
import json

subset_folder = "/afs/crc.nd.edu/user/s/swebster/Private/SecureSoftwareEngineering/analyze/subsets"
subset_files = [
    "baseline",
    "concise",
    "verbose"
]

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

evaluations = {f: load_json(os.path.join(subset_folder, f+".json")) for f in subset_files}

for key, data in evaluations.items():
    print(key)

    jailbreaking = data["tensortrust"]
    extraction = data["tt_extraction"]
    injection = []
    safe = []

    for ds in [data["train_deepset"], data["test_deepset"]]:
        for entry in ds:
            if entry["label"] == 0:
                safe.append(entry)
            else:
                injection.append(entry)

    # By attack type
    # for att in [jailbreaking, extraction, injection, safe]:
    #     n_accepted = len([e for e in att if e["accept"] == 1])
    #     n_rejected = len([e for e in att if e["accept"] == 0])
    #     print(n_accepted, n_rejected)

    # tp = 0
    # tn = 0
    # fp = 0
    # fn = 0
    # for att in [jailbreaking, extraction, injection, safe]:
    #     for entry in att:
    #         if entry["label"] == 1:
    #             if entry["accept"] == 0:
    #                 tp += 1
    #             else:
    #                 fn += 1
    #         elif entry["label"] == 0:
    #             if entry["accept"] == 0:
    #                 fp += 1
    #             else:
    #                 tn += 1
    
    # print(tp, tn, fp, fn)

    # By attack type
    for att in [jailbreaking, extraction, injection, safe]:
        n_leak = len([e for e in att if e["safety_mentioned"] == 1])
        n_no_leak = len([e for e in att if e["safety_mentioned"] == 0])
        print(n_leak, n_no_leak)