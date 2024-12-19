import json
import re
import os
import time

def load_data():
    # Tensor Trust
    with open("../tt_extraction.json") as f:
        tensortrust = json.load(f)
    return tensortrust

    with open('../tensortrust.json') as f:
        tensortrust = json.load(f)

    # Train Deepset
    with open('../train_deepset.json') as f:
        train_deepset = json.load(f)

        for entry in train_deepset:
            del entry["label"]
    
    # Test Deepset
    with open('../test_deepset.json') as f:
        test_deepset = json.load(f)

        for entry in test_deepset:
            del entry["label"]

    return tensortrust, train_deepset, test_deepset


def load_words():
    lists_dir = "./sublists"

    words = set()

    for filename in os.listdir(lists_dir):
        curr_list = os.path.join(lists_dir, filename)

        with open(curr_list, "r") as file:
            words.update({line.strip() for line in file})
    
    words = [(word, re.compile(r'\b({0})\b'.format(word), re.IGNORECASE)) for word in words]


    # words_list_file = "./google_blacklist.txt"

    # with open(words_list_file, "r") as file:
    #     # words = [line.strip() for line in file]
    #     # words = [(line.strip(), re.compile(r'\b({0})\b'.format(line.strip()), flags=re.IGNORECASE)) for line in file]

    #     words = [(line.strip(), re.compile(r'\b({0})\b'.format(line.strip()), re.IGNORECASE)) for line in file]
    
    return words

words = load_words()
print("Num blacklisted words:", len(words))
# tensortrust, train_deepset, test_deepset = load_data()
# datasets = [tensortrust, train_deepset, test_deepset]
tensortrust = load_data()
datasets = [tensortrust]
print("Num prompts: ", sum([len(ds) for ds in datasets]))
print("All data loaded!")

def countWholeWord(word, text):
    # Use re.findall to count occurrences of the whole word
    matches = re.findall(word, text, flags=re.IGNORECASE)
    return len(matches)

def get_blacklisted_words(prompt):
    found_words = dict()

    for word in words:
        ct = len(word[1].findall(prompt))
    
        if ct:
            found_words[word[0]] = ct
    
    return found_words

# @profile
def main():
    # start_time = time.perf_counter()
    for dataset in datasets:
        for entry in dataset:
            entry["blacklisted_words"] = get_blacklisted_words(entry["prompt"])
            
    # total_time = time.perf_counter() - start_time

    # print(f"Total time taken: {total_time}")
    # print(f"Avg time: {total_time/sum([len(ds) for ds in datasets])}")

    # # When testing with Scalene, don't write new files
    # exit()

    # Tensor Trust
    for entry in tensortrust:
        del entry["prompt"]
    with open("tt_extraction_blacklisted_words.json", 'w') as f:
        json.dump(tensortrust, f, indent=None)
    exit()

    # Train Deepset
    for entry in train_deepset:
        del entry["prompt"]
    with open("train_deepset_blacklisted_words.json", 'w') as f:
        json.dump(train_deepset, f, indent=None)

    # Test Deepset
    for entry in test_deepset:
        del entry["prompt"]
    with open("test_deepset_blacklisted_words.json", 'w') as f:
        json.dump(test_deepset, f, indent=None)
    


if __name__ == "__main__":
    main()