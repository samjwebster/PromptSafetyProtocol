import json

def compose_json(dataset):
    json_files = {
        "prompts": f"/afs/crc.nd.edu/user/s/swebster/Private/SecureSoftwareEngineering/data/{dataset}.json",
        "spam": f"/afs/crc.nd.edu/user/s/swebster/Private/SecureSoftwareEngineering/data/spam_eval/{dataset}_spam_eval.json",
        "blacklisted": f"/afs/crc.nd.edu/user/s/swebster/Private/SecureSoftwareEngineering/data/blacklisted_words/{dataset}_blacklisted_words.json",
        "tweeteval": f"/afs/crc.nd.edu/user/s/swebster/Private/SecureSoftwareEngineering/tweeteval/{dataset}_tweeteval.json"
    }

    full_data = dict()

    def get_data(json_file):
        with open(json_file) as f:
            return json.load(f)

    # Prompts
    data = get_data(json_files["prompts"])
    for entry in data:
        # TensorTrust doesn't have labels, since all are attacks. Add that for simplicity later
        label = 1
        if "label" in entry.keys():
            label = entry["label"]

        full_data[entry["id"]] = {
            "id": entry["id"],
            "prompt": entry["prompt"],
            "label": label
        }

    # Spam Evaluation
    data = get_data(json_files["spam"])
    for entry in data:
        full_data[entry["id"]]["spam"] = {
            "ngram": entry["ngram"],
            "entropy": entry["entropy"]
        }
    
    # Blacklisted Count
    data = get_data(json_files["blacklisted"])
    for entry in data:
        full_data[entry["id"]]["blacklisted"] = entry["blacklisted_words"]
    
    # TweetEval
    data = get_data(json_files["tweeteval"])        
    for entry in data:
        full_data[entry["id"]]["tweeteval"] = {
            "emotion": sorted(entry["emotion"], key=lambda d: d["score"], reverse=True),
            "irony": sorted(entry["irony"], key=lambda d: d["score"], reverse=True),
            "offensive": sorted(entry["offensive"], key=lambda d: d["score"], reverse=True),
            "hate": sorted(entry["hate"], key=lambda d: d["score"], reverse=True),
            "sentiment": sorted(entry["sentiment"], key=lambda d: d["score"], reverse=True)
        }
    
    outfile = f"./{dataset}_full.json"
    with open(outfile, 'w') as f:
        json.dump(full_data, f, indent=None)

def main():
    # datasets = ["tensortrust", "train_deepset", "test_deepset"]
    datasets = ["tt_extraction"]
    for dataset in datasets:
        compose_json(dataset)

if __name__ == "__main__":
    main()