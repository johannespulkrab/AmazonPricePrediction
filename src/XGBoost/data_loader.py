import glob
import pickle
from scipy.sparse import hstack, vstack, csr_matrix, load_npz, save_npz
import numpy as np


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_text_feature_batches(processed_dir="data/processed", batch_pattern='text_features_batch_*.pkl'):
    batch_files = glob.glob(f"{processed_dir}/{batch_pattern}")
    batch_files.sort()
    
    all_batches = []
    for file_path in batch_files:
        batch_data = load_pickle(file_path)
        all_batches.append(batch_data)
    
    # Combine all batches
    combined = vstack(all_batches)
    return combined


def load_amazon_data(processed_dir="data/processed", 
                    metadata_file='metadata.pkl',
                    numerical_file='numerical_features.pkl',
                    text_batch_pattern='text_features_batch_*.pkl'):
    
    # Load metadata
    metadata_path = f"{processed_dir}/{metadata_file}"
    metadata = load_pickle(metadata_path)
    
    # Load numerical features
    numerical_path = f"{processed_dir}/{numerical_file}"
    numerical_features = load_pickle(numerical_path)

    # Load text features
    text_features = load_text_feature_batches(processed_dir, text_batch_pattern)

    return {
        'metadata': metadata,
        'numerical_features': numerical_features,
        'text_features': text_features
    }


def combine_features(numerical_features, text_features, metadata):
    y = metadata.get('y', None)
    if y is None:
        raise ValueError('Target variable y not found in metadata')
    
    # Combine features using scipy's hstack
    X = hstack([numerical_features, text_features])
    
    # Check dimensions
    if X.shape[0] != len(y):
        raise ValueError(f"Feature matrix shape {X.shape} does not match target vector shape {y.shape}")
    
    return X, y


def load_and_prepare_data(processed_dir="data/processed",
                         force_rebuild=False,
                         assembled_path="data/processed/X_all_xgb.npz"):
    
    try:
        if not force_rebuild:
            X = load_npz(assembled_path)
            # Still need metadata for y
            data = load_amazon_data(processed_dir)
            metadata = data['metadata']
            y = metadata['y']
        else:
            raise FileNotFoundError
    except (FileNotFoundError, ValueError):
        print("Assembling features...")
        # Load all data components
        data = load_amazon_data(processed_dir)
        
        # Combine features and get target
        X, y = combine_features(
            data['numerical_features'], 
            data['text_features'],  
            data['metadata']
        )
        
        # Save assembled features
        save_npz(assembled_path, X)
        print(f"Features assembled and saved to {assembled_path}")
    
    return X, y


if __name__ == "__main__":
    X, y = load_and_prepare_data(force_rebuild=False)
    print("---Features matrix shape---")
    print(X.shape)
    print("---Target vector length---")
    print(len(y))