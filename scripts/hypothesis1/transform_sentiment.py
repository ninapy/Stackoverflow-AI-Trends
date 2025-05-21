import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

CURR_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TO_CSV = os.path.join(CURR_PATH, '../..', 'data/hypothesis1_with_frustration_scores.csv')

file = pd.read_csv(PATH_TO_CSV)
df = pd.DataFrame(file)
print(df.columns)
scaler = MinMaxScaler()
df['minmax_sentiment'] = df.groupby('language_type')['sentiment'].transform(
    lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
    )

df.to_csv('data/minmax_sentiment.csv')