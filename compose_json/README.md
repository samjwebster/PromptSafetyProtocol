# `/compose_json`

This folder is where the computed prompt-specific safety metrics are composed from their various locations in `/data` and `/tweeteval`. 

`compose_json.py`: Gets the computed safety metrics for all prompts for all datasets and puts them together into master 'full' json files for each used dataset.

`*_full.json`: Each dataset of prompts, each prompt with all of its computed safety features.