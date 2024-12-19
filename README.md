# README.md

This is the base directory. If you go into any other folder, it will have its own README explaining the purpose of it and its files.

## How to run:
1. Create the environment: `conda create --name sse --file req.txt`
2. Activate the environment: `conda activate sse`
3. Run the `process_prompts.py <verbose> <tag> <reject mode> <dataset>` scripts with whatever arguments you'd like. Recommended example: `process_prompts.py False test-run none mini`. This runs the 'mini' test (containing 4 example prompts, one from each dataset/class) with a concise prompt safety report and no rejection mechanism, with the file tag 'test-run'. Note: it's recommended you run this on a machine with a GPU since it involves running an LLM. Provided is `submit.sh` which is a job script I used on CRC to run this.
4. View the newly created file based on your specifications, which will contain the LLM responses to each prompt.

Contents:
`req.txt`: Conda environment requirements file.

`safety_protocol.py`: Contains the Python class implementation for the prompt safety protocol. When provided the prompt and its processed data, it returns the pre-prompt safety context, the post-prompt safety report, and rejection decisions if necessary. The class is primarily configurable to the explored concise/verbose report modes as well rejection mechanism modes. The rejection threshold can also be overridden as well as security feature weights, though those features were not explored due to the lack of systematic way to explore them.

`process_prompts.py`: Python code that accepts a safety protocol configuration as well as a file containing which prompts to assess. This is what's used to create LLM outputs to assess the prompt safety protocol's performance.

`submit.sh`: CRC job submitting file for running `process_prompts.py`.

`process_baseline.py`: A modified version of `process_prompts.py` that removes the prompt safety protocol functionality for creation of the baseline LLM response data.

`get_thresholds.py`: A script that uses the prompt safety protocol to view rejection scores and compute EER thresholds to be used as default weights for both rejection mechanism modes.