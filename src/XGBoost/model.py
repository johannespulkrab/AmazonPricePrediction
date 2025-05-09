''' XGBoost model definition.'''

import logging
import numpy as np
import xgboost as xgb
import os
import pickle
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("*" * 50)
print("XGBoost MODEL SCRIPT STARTED")
print("*" * 50)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XGBoostRegressor:
    def __init__(self, params: Optional [Dict[str, Any]] = None):
        self.default_params = {
            'objective': 'reg.squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'n_estimators': 100,
            'random_state': 42
        }

        self.params = params if params is not None else self.default_params
        self.model = None

    def train(self, X_train, y_train, X_val = None, y_val = None, early_stopping_rounds = 30, verbose = True):
        logger.info('Model training start')

        dtrain = xgb.DMatrix(X_train, label=y_train) # XGBoost's specialized data structure for input data
        watchlist = [(dtrain, 'train')]

        eval_result = {}

        dval = xgb.DMatrix(X_val, label=y_val)
        watchlist = [(dval, 'eval')]

        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.params.get('n_estimators', 100),
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
            eval_results=eval_result,
            verbose_eval=10 if verbose else 0
        )

        logger.info('Model training completed')

        results = {
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else None,
            'best_score': self.model.best_score if hasattr(self.model, 'best_score') else None,
            'train_rmse': eval_result.get('train', {}).get('rmse', [])[-1] if 'train' in eval_result else None,
            'val_rmse': eval_result.get('eval', {}).get('rmse', [])[-1] if 'eval' in eval_result else None
        }

        logger.info(f'Training results {results}')
        return results
    

    def predict(self, X):

        ''' X: features to predict on'''

        if self.model is None:
            raise ValueError('Model has not been trained') 
        
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def evaluate(self, X, y_true):
        if self.model is None:
            raise ValueError('Model has not been trained') 
        
        y_pred = self.predict(X)

        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_pred, y_true)
        }

        logger.info(f'Model evaluation metrics {metrics}')
        return metrics
    
    def save_model(self, file_path):
        if self.model is None:
            raise ValueError('Model has not been trained') 
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        self.model.save_model(file_path)

        params_path = file_path + '.params'
        with open(params_path, 'wb') as f:
            pickle.dump(self.params, f)
        logger.info(f'Model params are saved to {params_path}')

    
    def load_model(self, file_path):
        self.model = xgb.Booster()
        self.model.load_model(file_path)

        params_path = file_path + '.params'
        try:
            with open(params_path, 'rb') as f:
                self.params = pickle.load(f)
            logger.info(f"Model parameters loaded from {params_path}")
        except FileNotFoundError:
            logger.warning(f"Model parameters file not found at {params_path}")


# For testing

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test XGBoost model for Amazon price prediction')
    parser.add_argument('--train_data', type=str, help='Path to training data pickle file')
    parser.add_argument('--model_output', type=str, default='./models/xgboost_model.json', 
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    if args.train_data:
        try:
            with open(args.train_data, 'rb') as f:
                data = pickle.load(f)
            
            X_train = data.get('X_train')
            y_train = data.get('y_train')
            X_val = data.get('X_val')
            y_val = data.get('y_val')
            
            if X_train is not None and y_train is not None:
                model = XGBoostRegressor()
                model.train(X_train, y_train, X_val, y_val)
                
                model.save_model(args.model_output)
                logger.info(f"Model saved to {args.model_output}")
            else:
                logger.error("Training data does not contain X_train and y_train")
        except Exception as e:
            logger.error(f"Error in model testing: {e}")
    else:
        logger.info("No training data provided. Skipping model training test.")


