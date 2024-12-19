import pandas as pd

splits = {'train': 'data/train-00000-of-00001-9564e8b05b4757ab.parquet', 'test': 'data/test-00000-of-00001-701d16158af87368.parquet'}
df = pd.read_parquet("hf://datasets/deepset/prompt-injections/" + splits["train"])

print(df.head(5))