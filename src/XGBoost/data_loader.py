''' Handles loading preprocessed data.'''

import os
import pickle
import logging
import numpy as np
import glob
from typing import Tuple, Dict, Any, Optional


print("*" * 50)
print("DATA LOADER SCRIPT STARTED")
print("*" * 50)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

def load_pickle(file_path: str) -> Any:
    logger.info(f'Loading data {file_path}')
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info('Data loaded successfully')
        return data
    except Exception as e:
        logger.info(f"Error loading data from {file_path}: {e}")
        raise



def load_text_feature_batches(processed_dir: str, batch_pattern: str = 'text_features_batch_*.pkl') -> Dict[int, Any]:
    logger.info(f'Loading data {processed_dir}')
    batches = {}

    try:
        batch_files = glob.glob(os.path.join(processed_dir, batch_pattern)) # Finds files matching pattern
        batch_files.sort()

        logger.info(f'Found {len(batch_files)} text feature batch files')

        for file_path in batch_files:
            batch_id = int(os.path.basename(file_path).split('_')[-1].split('.')[0])
            batches[batch_id] = load_pickle(file_path)

        return batches
    except Exception as e:
        logger.error(f'Error {e}')
        raise


def load_amazon_data(processed_dir: str, 
                    metadata_file: str = 'metadata.pkl',
                    numerical_file: str = 'numerical_features.pkl',
                    text_batch_pattern: str = 'text_features_batch_*.pkl',
                    max_batches: Optional[int] = None) -> Dict[str, Any]:
    logger.info(f'Loading all files from {processed_dir}')

    metadata_path = os.path.join(processed_dir, metadata_file)
    metadata = load_pickle(metadata_path)

    numerical_path = os.path.join(processed_dir, numerical_file)
    numerical_features = load_pickle(numerical_path)

    text_batches = load_text_feature_batches(processed_dir, text_batch_pattern)


    if max_batches is not None and max_batches > 0:
        batch_ids = sorted(list(text_batches.keys()))[:max_batches]
        text_batches = {k: text_batches[k] for k in batch_ids}
        logger.info(f"Limited to {len(text_batches)} text feature batches for debugging")
    
    return {
        'metadata': metadata,
        'numerical_features': numerical_features,
        'text_batches': text_batches
    }

def combine_features(numerical_features: np.ndarray, 
                     text_batches: Dict[int, np.ndarray],
                     metadata: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    logger.info('Executin the combination of numerical and text features')
    
    y = metadata.get('y', None)
    if y is None:
        raise ValueError('Target variable y not found in metadata')
    
    text_features = []
    batch_indices = sorted(text_batches.keys())

    for batch_id in batch_indices:
        text_features.append(text_batches[batch_id])

    combined_text = np.vstack(text_features)
    logger.info(f"Combined text features shape: {combined_text.shape}")

    if numerical_features.ndim == 1:
        numerical_features = numerical_features.reshape(-1, 1)

    if combined_text.ndim == 1:
        combined_text = combined_text.reshape(-1, 1)

    # Combination of numerical and text features 
    X = np.hstack([numerical_features, combined_text])
    logger.info(f"Final combined features shape: {X.shape}")

    if len(X) != len(y):
        raise ValueError(F'Feature matrix shape {X.shape} does not match target verctor shape {y.shape}')
    
    return X, y

def split_data(X: np.ndarray, y: np.ndarray, 
               train_ratio: float = 0.7, 
               val_ratio: float = 0.15, 
               test_ratio: float = 0.15, 
               random_state: int = 42) -> Dict[str, np.ndarray]:
    
    ''' Data split into train, val and test'''

    logger.info(f"Splitting data with ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    np.random.seed(random_state)

    indices = np.random.permutation(len(X))
    train_end = int(train_ratio * len(X))
    val_end = train_end + int(val_ratio * len(X))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    logger.info(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }

def load_and_prepare_data(processed_dir: str,
                         train_ratio: float = 0.7, 
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15,
                         random_state: int = 42,
                         max_batches: Optional[int] = None) -> Dict[str, np.ndarray]:

    data = load_amazon_data(processed_dir, max_batches=max_batches)

    X, y = combine_features(data['numerical_features'], data['text_batches'], data['metadata'])

    data_splits = split_data(X, y, train_ratio, val_ratio, test_ratio, random_state)

    return data_splits

if __name__ == "__main__":
    print("Starting data loader...")
    # Testing the data loader
    import argparse
    
    parser = argparse.ArgumentParser(description='Load and prepare data for Amazon price prediction')
    parser.add_argument('--processed_dir', type=str, required=True, help='Directory containing processed data files')
    parser.add_argument('--max_batches', type=int, default=None, help='Maximum number of text batches to load')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Proportion of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Proportion of data for validation')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Proportion of data for testing')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    data = load_and_prepare_data(
        args.processed_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
        max_batches=args.max_batches
    )
    
    logger.info("Data preparation completed successfully")
    logger.info(f"Available splits: {list(data.keys())}")
    for key, value in data.items():
        if hasattr(value, 'shape'):
            logger.info(f"{key} shape: {value.shape}")