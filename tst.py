import pandas as pd

df = pd.read_csv('./data_file/data.csv')
assets = df['tic'].unique()
print(1 + len(assets) + len(assets) * 5)
