from pathlib import Path

import pandas as pd
import numpy as np
import spacy
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


csv_path = Path('data/cleaned_amz_us_price_prediction_dataset.csv')


nlp = spacy.load("en_core_web_sm")



class AmazonDataset:
    def __init__(self, path):
        with open(path, 'r') as file:
            self.df = pd.read_csv(file)

        unique_categories = self.df['category'].unique()
        self.cat2id = {cat: idx for idx, cat in enumerate(unique_categories)}
        self.cat_encoder = OneHotEncoder(sparse_output=False)
        cat_encoded = self.cat_encoder.fit_transform(self.df[['category']])
        best_encoded = self.df['isBestSeller'].astype(int).values.reshape(-1, 1)

        self.numerical_features = np.hstack((cat_encoded, best_encoded))

        self.titles = self.df['title'].apply(self.preprocess_text).tolist()

        self.tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.8)
        self.text_features = self.tfidf.fit_transform(self.titles)


        self.X = self.combine_features()

    def preprocess_text(self, text):
        doc = nlp(text)
        tokens = [token.lemma_.lower() for token in doc
                  if not token.is_punct and not token.is_space and not token.is_stop]
        return ' '.join(tokens)

    def combine_features(self):

        text_features_dense = self.text_features.toarray()

        return np.hstack((text_features_dense, self.numerical_features))

if __name__ == '__main__':
    data = AmazonDataset(csv_path)
    print(data.X)
