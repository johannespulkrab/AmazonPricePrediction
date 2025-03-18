from pathlib import Path
import pandas as pd
import spacy
from sklearn.preprocessing import OneHotEncoder
from sklearn import __version__ as sklearn_version
from packaging.version import Version
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from tqdm import tqdm
import pickle
from scipy import sparse

csv_path = Path('data/cleaned_amz_us_price_prediction_dataset.csv')

nlp = spacy.load("en_core_web_sm")


class AmazonDataset:
    def __init__(self, path, use_hashing=True, n_features=100000, batch_size=10000):
        print("Daten werden geladen...")
        with open(path, 'r') as file:
            self.df = pd.read_csv(file)

        self.batch_size = batch_size
        self.y = self.df['price'].values

        print("Kategorien werden verarbeitet...")
        unique_categories = self.df['category'].unique()
        self.cat2id = {cat: idx for idx, cat in enumerate(unique_categories)}
        if Version(sklearn_version) >= Version('1.2.0'):
            self.cat_encoder = OneHotEncoder(sparse_output=True)
        else:
            self.cat_encoder = OneHotEncoder(sparse=True)  # ältere Versionen
        cat_encoded = self.cat_encoder.fit_transform(self.df[['category']])
        best_encoded = sparse.csr_matrix(self.df['isBestSeller'].astype(int).values.reshape(-1, 1))

        self.numerical_features = sparse.hstack((cat_encoded, best_encoded))

        print("Texte werden vorverarbeitet...")
        self.titles = []
        for title in tqdm(self.df['title'], desc="Textvorverarbeitung"):
            self.titles.append(self.preprocess_text(title))

        print("Text-Vektorisierung wird durchgeführt...")
        if use_hashing:
            self.vectorizer = HashingVectorizer(n_features=n_features, ngram_range=(1, 2))
            self.text_features = self.vectorizer.transform(self.titles)
        else:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.5,
                max_features=n_features
            )
            self.text_features = self.vectorizer.fit_transform(self.titles)

        print(f"Text Features Shape: {self.text_features.shape}")
        print(f"Numerical Features Shape: {self.numerical_features.shape}")


        print("Verarbeitung abgeschlossen!")

    def preprocess_text(self, text):
        doc = nlp(text)
        tokens = [token.lemma_.lower() for token in doc
                  if not token.is_punct and not token.is_space and not token.is_stop]
        return ' '.join(tokens)

    def save_features(self, output_path):
        print("Speichere Features...")

        n_samples = self.text_features.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        for i in tqdm(range(n_batches), desc="Speichere Text-Features"):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)

            batch_features = self.text_features[start_idx:end_idx]

            with open(f'{output_path}/text_features_batch_{i}.pkl', 'wb') as f:
                pickle.dump(batch_features, f)

        # Speichere numerische Features
        with open(f'{output_path}/numerical_features.pkl', 'wb') as f:
            pickle.dump(self.numerical_features, f)

        # Speichere Metadaten
        with open(f'{output_path}/metadata.pkl', 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'cat_encoder': self.cat_encoder,
                'cat2id': self.cat2id,
                'n_samples': n_samples,
                'n_batches': n_batches,
                'batch_size': self.batch_size,
                'y': self.y  # Preise speichern
            }, f)

        print(f"Features wurden in {output_path} gespeichert!")


if __name__ == '__main__':
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True, parents=True)

    data = AmazonDataset(csv_path, use_hashing=True, n_features=100000, batch_size=10000)
    data.save_features(output_dir)

    print("Verarbeitung abgeschlossen!")