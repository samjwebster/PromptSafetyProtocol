import json
import os
import math
from collections import Counter
import cProfile

def load_data():
    with open('./tt_extraction.json') as f:
        tensortrust = json.load(f)
    return tensortrust

    # Tensor Trust
    with open('./tensortrust.json') as f:
        tensortrust = json.load(f)

    # Train Deepset
    with open('./train_deepset.json') as f:
        train_deepset = json.load(f)

        for entry in train_deepset:
            del entry["label"]
    
    # Test Deepset
    with open('./test_deepset.json') as f:
        test_deepset = json.load(f)

        for entry in test_deepset:
            del entry["label"]

    return tensortrust, train_deepset, test_deepset

def evaluate_ngram(n, text):
    ngram_frequencies = dict()

    assert len(text) >= n

    for i in range(len(text) - n + 1):
        ngram = text[i:i + n]
        if ngram in ngram_frequencies:
            ngram_frequencies[ngram] += 1
        else:
            ngram_frequencies[ngram] = 1

    ngram_frequencies = dict(reversed(sorted(ngram_frequencies.items(), key=lambda item: item[1])))

    return ngram_frequencies

def analyze_freqs(freqs):
    top_five = list(freqs.keys())[:3]
    total_grams = sum(freqs.values())

    top_weight = 0.0
    for key in top_five:
        top_weight += freqs[key]/total_grams

    return top_weight

def calculate_entropy(sequence):
    token_counts = Counter(sequence)
    total_tokens = len(sequence)
    
    entropy = 0
    for count in token_counts.values():
        probability = count / total_tokens
        entropy -= probability * math.log2(probability)
    return entropy

def normalized_entropy(sequence):
    actual_entropy = calculate_entropy(sequence)
    max_entropy = math.log2(len(sequence)) if len(sequence) > 1 else 1  # Avoid log(0) when len = 1
    
    return actual_entropy / max_entropy

def get_ngram_and_entropy(text):
    avg_weight = 0
    n_weights = 0
    for i in range(1, 11):
        try:
            curr_freqs = evaluate_ngram(i, text)
            curr_weight = analyze_freqs(curr_freqs)
            avg_weight += curr_weight
            n_weights += 1
        except:
            break

    ngram = avg_weight/n_weights

    entropy = normalized_entropy(list(text.replace(' ', '')))

    return ngram, entropy

def main():

    # tensortrust, train_deepset, test_deepset = load_data()
    # datasets = [tensortrust, train_deepset, test_deepset]
    tensortrust = load_data()
    datasets = [tensortrust]

    for dataset in datasets:
        for entry in dataset:
            ngram, entropy = get_ngram_and_entropy(entry["prompt"])
            entry["ngram"] = ngram
            entry["entropy"] = entropy

    # Tensor Trust
    for entry in tensortrust:
        del entry["prompt"]
    with open("tt_extraction_spam_eval.json", 'w') as f:
        json.dump(tensortrust, f, indent=None)

    # with open("tensortrust_spam_eval.json", 'w') as f:
    #     json.dump(tensortrust, f, indent=None)

    # Train Deepset
    for entry in train_deepset:
        del entry["prompt"]
    with open("train_deepset_spam_eval.json", 'w') as f:
        json.dump(train_deepset, f, indent=None)

    # Test Deepset
    for entry in test_deepset:
        del entry["prompt"]
    with open("test_deepset_spam_eval.json", 'w') as f:
        json.dump(test_deepset, f, indent=None)

if __name__ == "__main__":
    main()