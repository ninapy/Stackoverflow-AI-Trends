import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ttest_ind
import seaborn as sns

CURR_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TO_CSV = os.path.join(CURR_PATH, '../..', 'data/minmax_sentiment.csv')

df = pd.read_csv('data/minmax_sentiment.csv')
frustration_high_level = df[df['language_type'] == 'high-level']['minmax_sentiment']
frustration_low_level = df[df['language_type'] == 'low-level']['minmax_sentiment']

truncated_frust_high = frustration_high_level[frustration_high_level >= 0.95]
truncated_frust_low = frustration_low_level[frustration_low_level >= 0.95]

def plot_violin():
    sns.violinplot(data=df, x='language_type', y='minmax_sentiment')
    plt.title('Frustration density by language')
    plt.xticks(rotation=45)
    plt.show()

def plot_hist_high_end():
    sns.histplot(truncated_frust_high, bins=30, color='blue', label='High-level frustration', kde=True)
    sns.histplot(truncated_frust_low, bins=30, alpha=0.6, color='orange', label='Low-level frustration', kde=True)
    plt.title('Frustration Score Distribution by Language Type')
    plt.legend()
    plt.show()

t_stat, p_value = ttest_ind(frustration_high_level, frustration_low_level, equal_var=False)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
plot_violin()
plot_hist_high_end()