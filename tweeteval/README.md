# `/tweeteval`

This folder is originally the cloned git repository for TweetEval, used for the qualitative measurements in the safety report. I've added the following files:

`*_tweeteval.json`: Each dataset with computed TweetEval qualitative results for each prompt.

`evaluate_prompts.py`: Processes all datasets over all qualitative tasks to craete the *_tweeteval.json files. 

`submit.sh`: File for submitting a CRC job to run the evaluate_prompts.py script. Since this utilizes classifier models, it's much faster to do on GPU.

`logs/`: For holding any logs created by the submit.sh jobs.

Other original TweetEval files:
- `datasets/`
- `predictions/`
- `tweeteval_README.md`
- `TweetEval_Tutorial.ipynb`
- `evaluation_script.py`