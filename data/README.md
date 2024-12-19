# `/data`

This folder contains two main things: the original dataset files and the quantitative safety analyses. 

`*.json`: Each original dataset, with each entry containing a prompt, an ID, and a label where 0 is safe and 1 is malicious.

`blacklisted_words/`: Contains necessary files for computing the blacklisted words present in each prompt, as well as the computed JSONs containing results for each dataset. 
- `blacklisted_words/*_blacklisted_words.json`: Each dataset with all blacklisted words present in each prompt. Most prompts are free of any blacklisted words.
- `blacklisted_words/sublists/*.txt/`: Files containing the blacklisted words sublists from CMU, Google, Openverse, and Shutterstock.
- `count_blacklisted_words.py`: Uses the original prompts and the blacklisted subsets to create the *_blacklisted_words.json files.

`spam_eval/`: Contains necessary files for computing the spam evaluation for each prompt, as well as the computed JSONs containing results for each dataset.
- `spam_eval/spam_eval.py`: Performs the n-gram and entropy based computations for spam evaluation and writes the results to new JSON files.
- `spam_eval/*_spam_eval.json`: Each dataset with computed spam evaluation for each prompt.

`retrieve_data/`: For the TensorTrust and Deepset datasets, scripts were needed in order to download them. This folder contains those scripts.