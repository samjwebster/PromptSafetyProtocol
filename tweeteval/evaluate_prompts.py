from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import json
import time
import sys

def load_data():
    # Tensor Trust
    with open('../data/tt_extraction.json') as f:
        tensortrust = json.load(f)

    return tensortrust

    with open('../data/tensortrust.json') as f:
        tensortrust = json.load(f)

    # Train Deepset
    with open('../data/train_deepset.json') as f:
        train_deepset = json.load(f)

        for entry in train_deepset:
            del entry["label"]
    
    # Test Deepset
    with open('../data/test_deepset.json') as f:
        test_deepset = json.load(f)

        for entry in test_deepset:
            del entry["label"]

    return tensortrust, train_deepset, test_deepset

def load_model_and_labels(task):
    # Load model
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Download label mapping
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    return model, tokenizer, labels

# Load the json files
# tensortrust, train_deepset, test_deepset = load_data()
# datasets = [tensortrust, train_deepset, test_deepset]
tensortrust = load_data()
datasets = [tensortrust]
print("Data loaded successfully.")

def main():
    tasks = [
        "emotion", # Emotion detection
        "hate", # Hate speech detection
        "irony", # Irony detection
        "offensive", # Offensive language detection
        "sentiment", # Sentiment analysis
    ]

    # tasks = [sys.argv[1]]
    # print(tasks)
    # exit()

    for task in tasks:

        print(f"Evaluating: {task}")

        model, tokenizer, labels = load_model_and_labels(task)

        def analyze_prompt(prompt):
            encoded_input = tokenizer(prompt, truncation=True, max_length=512, return_tensors='pt')
            output = model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)

            ranking = np.argsort(scores)
            ranking = ranking[::-1]

            analyzed = []
            for i in range(scores.shape[0]):
                l = labels[ranking[i]]
                s = float(scores[ranking[i]])
                analyzed.append({"label": l, "score": s})

            return analyzed

        # start_time = time.perf_counter()

        for i, dataset in enumerate(datasets):
            print(f"Dataset {i}...")

            for entry in dataset:
                entry[task] = analyze_prompt(entry["prompt"])

        # total_time = time.perf_counter() - start_time
        # print(f"Time taken: {total_time}")
        # print(f"Average time: {total_time/sum([len(ds) for ds in datasets])}")

    # Tensortrust
    for entry in tensortrust:
        del entry["prompt"]
    with open("tt_extraction_tweeteval.json", 'w') as f:
        json.dump(tensortrust, f, indent=None)

    # # Train Deepset
    # for entry in train_deepset:
    #     del entry["prompt"]
    # with open("train_deepset_tweeteval.json", 'w') as f:
    #     json.dump(train_deepset, f, indent=None)

    # # Test Deepset
    # for entry in test_deepset:
    #     del entry["prompt"]
    # with open("test_deepset_tweeteval.json", 'w') as f:
    #     json.dump(test_deepset, f, indent=None)
    
    print("All files saved!")


if __name__ == "__main__":
    main()