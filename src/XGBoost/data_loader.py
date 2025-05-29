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



def load_text_feature_batches(processed_dir: str, batch_pattern: str = 'text_features_batch_*.pkl') -> np.ndarray:
    logger.info(f'Loading data {processed_dir}')
    
    try:
        batch_files = glob.glob(os.path.join(processed_dir, batch_pattern)) 
        batch_files.sort()
        
        logger.info(f'Found {len(batch_files)} text feature batch files')
        
        # Load all batches
        all_batches = []
        total_rows = 0
        
        for file_path in batch_files:
            batch_id = int(os.path.basename(file_path).split('_')[-1].split('.')[0])
            batch_data = load_pickle(file_path)
            
            # Track number of rows
            rows = batch_data.shape[0]
            total_rows += rows
            
            logger.info(f"Loaded batch {batch_id}: shape={batch_data.shape}, rows={rows}")
            all_batches.append(batch_data)
        
        # Now use scipy's vstack for sparse matrices
        from scipy.sparse import vstack
        combined = vstack(all_batches)
        logger.info(f"Combined shape: {combined.shape}, total rows: {total_rows}")
        
        return combined
    except Exception as e:
        logger.error(f'Error: {e}')
        import traceback
        logger.error(traceback.format_exc())
        raise


def load_amazon_data(processed_dir: str, 
                    metadata_file: str = 'metadata.pkl',
                    numerical_file: str = 'numerical_features.pkl',
                    text_batch_pattern: str = 'text_features_batch_*.pkl'):
    logger.info(f'Loading all files from {processed_dir}')

    # Load and examine metadata first
    metadata_path = os.path.join(processed_dir, metadata_file)
    metadata = load_pickle(metadata_path)
    
    # Log the metadata structure
    logger.info(f"Metadata type: {type(metadata)}")
    if isinstance(metadata, dict):
        logger.info(f"Metadata keys: {list(metadata.keys())}")
        
        # Check the 'y' target value
        if 'y' in metadata:
            y_data = metadata['y']
            logger.info(f"Target 'y' type: {type(y_data)}")
            logger.info(f"Target 'y' shape or length: {y_data.shape if hasattr(y_data, 'shape') else len(y_data)}")
        
        # Check for batch information
        if 'n_batches' in metadata:
            logger.info(f"Metadata n_batches: {metadata['n_batches']}")
        if 'batch_size' in metadata:
            logger.info(f"Metadata batch_size: {metadata['batch_size']}")
    
    # Continue with loading other files
    numerical_path = os.path.join(processed_dir, numerical_file)
    numerical_features = load_pickle(numerical_path)

    text_features = load_text_feature_batches(processed_dir, text_batch_pattern)

    return {
        'metadata': metadata,
        'numerical_features': numerical_features,
        'text_features': text_features
    }
    

def combine_features(numerical_features, text_features, metadata: Dict[str, Any]):
    logger.info('Executing the combination of numerical and text features')
    
    y = metadata.get('y', None)
    if y is None:
        raise ValueError('Target variable y not found in metadata')
    
    # Debug info
    logger.info(f"Numerical features type: {type(numerical_features)}")
    logger.info(f"Numerical features shape: {numerical_features.shape}")
    logger.info(f"Text features type: {type(text_features)}")
    logger.info(f"Text features shape: {text_features.shape}")
    logger.info(f"Target y shape: {y.shape}")
    
    # Use scipy's hstack for sparse matrices
    from scipy.sparse import hstack
    
    # Ensure both matrices are sparse
    if not hasattr(numerical_features, 'tocsr'):
        from scipy.sparse import csr_matrix
        numerical_features = csr_matrix(numerical_features)
        logger.info("Converted numerical features to sparse matrix")
    
    if not hasattr(text_features, 'tocsr'):
        from scipy.sparse import csr_matrix
        text_features = csr_matrix(text_features)
        logger.info("Converted text features to sparse matrix")
    
    # Combine features using scipy's hstack
    logger.info("Combining features using scipy.sparse.hstack")
    X = hstack([numerical_features, text_features])
    logger.info(f"Final combined features shape: {X.shape}")
    
    # Check dimensions
    if X.shape[0] != len(y):
        raise ValueError(f"Feature matrix shape {X.shape} does not match target vector shape {y.shape}")
    
    return X, y

def split_data(X, y: np.ndarray, 
               train_ratio: float = 0.7, 
               val_ratio: float = 0.15, 
               test_ratio: float = 0.15, 
               random_state: int = 42) -> Dict[str, Any]:
    
    """Data split into train, val and test"""

    logger.info(f"Splitting data with ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    np.random.seed(random_state)

    # Get number of samples (use shape[0] instead of len for sparse matrix compatibility)
    n_samples = X.shape[0]
    
    # Create indices for the splits
    indices = np.random.permutation(n_samples)
    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create the splits
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    logger.info(f"Split sizes: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")
    
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
        random_state=args.random_state
    )
    
    logger.info("Data preparation completed successfully")
    logger.info(f"Available splits: {list(data.keys())}")
    for key, value in data.items():
        if hasattr(value, 'shape'):
            logger.info(f"{key} shape: {value.shape}")