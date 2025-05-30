import pickle
import numpy as np
from scipy.sparse import hstack, load_npz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def load_processed_data(processed_dir="data/processed"):
    """Load processed features and combine them for linear regression"""

    with open(f"{processed_dir}/metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)

    with open(f"{processed_dir}/numerical_features.pkl", 'rb') as f:
        numerical_features = pickle.load(f)

    try:
        X_combined = load_npz(f"{processed_dir}/X_all_baseline.npz")
        print("Loaded cached combined features")
    except FileNotFoundError:
        print("Loading and combining text features...")

        text_batches = []
        batch_files = sorted(Path(processed_dir).glob("text_features_batch_*.pkl"))

        for batch_file in batch_files:
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
                text_batches.append(batch_data)

        from scipy.sparse import vstack
        text_features = vstack(text_batches)

        print(f"Text features shape: {text_features.shape}")
        print(f"Numerical features shape: {numerical_features.shape}")

        X_combined = hstack([numerical_features, text_features])

        from scipy.sparse import save_npz
        save_npz(f"{processed_dir}/X_all_baseline.npz", X_combined)
        print(f"Combined features saved to X_all_baseline.npz")

    y = metadata['y']

    return X_combined, y


def prepare_data_splits(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """Split data into train/val/test sets and apply scaling"""

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state
    )

    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )

    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_val': X_val_scaled,
        'y_val': y_val,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'scaler': scaler
    }


if __name__ == "__main__":
    X, y = load_processed_data()
    data_splits = prepare_data_splits(X, y)

    print("Data loading completed successfully")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")