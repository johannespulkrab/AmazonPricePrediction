from pathlib import Path
import pandas as pd
import numpy as np
import spacy
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import pickle


csv_path = Path('data/cleaned_amz_us_price_prediction_dataset.csv')

nlp = spacy.load("en_core_web_sm")

class AmazonDataset:
    def __init__(self, path):
        print("Daten werden geladen...")
        with open(path, 'r') as file:
            self.df = pd.read_csv(file)

        print("Kategorien werden verarbeitet...")
        unique_categories = self.df['category'].unique()
        self.cat2id = {cat: idx for idx, cat in enumerate(unique_categories)}
        self.cat_encoder = OneHotEncoder(sparse_output=False)
        cat_encoded = self.cat_encoder.fit_transform(self.df[['category']])
        best_encoded = self.df['isBestSeller'].astype(int).values.reshape(-1, 1)

        self.numerical_features = np.hstack((cat_encoded, best_encoded))

        print("Texte werden vorverarbeitet...")
        # tqdm für die Textverarbeitung verwenden
        self.titles = []
        for title in tqdm(self.df['title'], desc="Textvorverarbeitung"):
            self.titles.append(self.preprocess_text(title))

        print("TF-IDF Vektorisierung wird durchgeführt...")
        self.tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.8)
        self.text_features = self.tfidf.fit_transform(self.titles)

        print("Features werden kombiniert...")
        self.X = self.combine_features()
        print("Verarbeitung abgeschlossen!")

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
    print(data.X.shape)
    with open('data/processed_dataset.pkl', 'wb') as f:
        pickle.dump({
            'X': data.X,
            'numerical_features': data.numerical_features,
            'text_features': data.text_features,
            'tfidf_vectorizer': data.tfidf,
            'cat_encoder': data.cat_encoder,
            'cat2id': data.cat2id
        }, f)