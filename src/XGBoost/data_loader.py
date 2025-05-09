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



def load_text_feature_batches(processed_dir: str, batch_pattern: str = 'text_features_batch_*.pkl') -> Dict[int, np.ndarray]:
    logger.info(f'Loading data {processed_dir}')
    batches = {}

    try:
        batch_files = glob.glob(os.path.join(processed_dir, batch_pattern)) 
        batch_files.sort()

        logger.info(f'Found {len(batch_files)} text feature batch files')

        # Debug the content of the first batch
        if batch_files:
            first_batch = load_pickle(batch_files[0])
            logger.info(f"First batch type: {type(first_batch)}")
            logger.info(f"First batch shape: {first_batch.shape if hasattr(first_batch, 'shape') else 'No shape attribute'}")

        # Load all batches
        all_batches = []
        for file_path in batch_files:
            batch_data = load_pickle(file_path)
            all_batches.append(batch_data)
            
        # Now combine all batches into a single array
        combined_text = np.vstack(all_batches)
        logger.info(f"Combined all batches directly: {combined_text.shape}")
        
        # Return as a single entry (not as a dict of batches)
        return combined_text
    except Exception as e:
        logger.error(f'Error {e}')
        raise


def load_amazon_data(processed_dir: str, 
                    metadata_file: str = 'metadata.pkl',
                    numerical_file: str = 'numerical_features.pkl',
                    text_batch_pattern: str = 'text_features_batch_*.pkl'):
    logger.info(f'Loading all files from {processed_dir}')

    metadata_path = os.path.join(processed_dir, metadata_file)
    metadata = load_pickle(metadata_path)

    numerical_path = os.path.join(processed_dir, numerical_file)
    numerical_features = load_pickle(numerical_path)

    # Now text_features is already combined
    text_features = load_text_feature_batches(processed_dir, text_batch_pattern)

    return {
        'metadata': metadata,
        'numerical_features': numerical_features,
        'text_features': text_features  # Changed from text_batches to text_features
    }

def combine_features(numerical_features: np.ndarray, 
                     text_features: np.ndarray,  # Changed from text_batches to text_features
                     metadata: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    logger.info('Executing the combination of numerical and text features')
    
    y = metadata.get('y', None)
    if y is None:
        raise ValueError('Target variable y not found in metadata')
    
    # Debug info for numerical features
    logger.info(f"Numerical features type: {type(numerical_features)}")
    logger.info(f"Numerical features shape: {numerical_features.shape}")
    
    # Debug info for text features
    logger.info(f"Text features type: {type(text_features)}")
    logger.info(f"Text features shape: {text_features.shape}")
    
    # Convert sparse matrix to dense if needed
    if hasattr(numerical_features, 'toarray'):
        numerical_features = numerical_features.toarray()
        logger.info("Converted sparse matrix to numpy array")
        logger.info(f"Numerical features shape after conversion: {numerical_features.shape}")
    
    # Attempt to combine
    logger.info(f"Attempting to combine arrays with shapes: {numerical_features.shape} and {text_features.shape}")
    X = np.hstack([numerical_features, text_features])
    logger.info(f"Final combined features shape: {X.shape}")
    
    # Check if X and y have matching first dimension
    if len(X) != len(y):
        raise ValueError(f"Feature matrix shape {X.shape} does not match target vector shape {y.shape}")
    
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
                         random_state: int = 42) -> Dict[str, np.ndarray]:
    # Load all data components
    data = load_amazon_data(processed_dir)
    
    # Combine features and get target
    X, y = combine_features(
        data['numerical_features'], 
        data['text_features'],  # Changed from text_batches to text_features
        data['metadata']
    )
    
    # Split data
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