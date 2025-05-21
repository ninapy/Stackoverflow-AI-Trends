import pandas as pd
import os

# TODO: →combine title + body of posts
# import dataset
CURR_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TO_CSV = os.path.join(CURR_PATH, '..', 'data/cleaned_joined.csv')

file = pd.read_csv(PATH_TO_CSV)
df = pd.DataFrame(file)
print(df.columns)

# drop duplicate rows
print('init shape', df.shape)
df = df.drop_duplicates()
print('after shape', df.shape)

# compute average sentiment per group (high/low level) of language level posts using ML sentiment analysis model
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

model.eval()
iter = 0
def get_sentiment(text):
    global iter

    input = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**input).logits

    probs = F.softmax(logits, dim=1)
    frustration_score = probs[0][model.config.label2id['NEGATIVE']].item()
    if iter % 10 == 0:
        print(f'{iter} rows processed')
    iter += 1
    return frustration_score


# create dataset with this data in data/hypothesis1_with_frustration_scores.csv
df['sentiment'] = df['full_text'].apply(get_sentiment)

OUTPUT_PATH = os.path.join(CURR_PATH, '..', 'data/hypothesis1_with_frustration_scores.csv')

df.to_csv(OUTPUT_PATH, index=False)