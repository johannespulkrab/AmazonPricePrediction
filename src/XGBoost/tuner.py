''' HP tuning'''

import logging
import numpy as np
import xgboost as xgb
import optuna
import pickle
import os
from typing import Dict, Any, Optional
from model import XGBoostRegressor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptunaHyperparameterTuner:

    def __init__(self,
                 param_space: Optional[Dict[str, Any]]=None,
                 n_trials=30,
                 timeout: Optional[int]=None,
                 study_name='xgboost_optimization',
                 direction='minimize',
                 metric='rmse'):
        self.param_space = param_space or self._default_param_space()
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name
        self.direction = direction
        self.metric = metric
        self.study = None
        self.best_params = None

        logger.info(f"Initialized Optuna tuner with {n_trials} trials, direction: {direction}, metric: {metric}")

    def default_param_space(self):

        return {
            'learning_rate': (0.01, 0.3),
            'max_depth': (3, 12),
            'min_child_weight': (1, 10),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'gamma': (0, 5),
            'alpha': (0, 10),
            'lambda': (0, 10),
            'n_estimators': (50, 500)
        }

    def _objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: np.ndarray, y_val: np.ndarray) -> float:
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': trial.suggest_float('learning_rate', 
                                              *self.param_space['learning_rate']),
            'max_depth': trial.suggest_int('max_depth', 
                                        *self.param_space['max_depth']),
            'min_child_weight': trial.suggest_int('min_child_weight', 
                                               *self.param_space['min_child_weight']),
            'subsample': trial.suggest_float('subsample', 
                                          *self.param_space['subsample']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 
                                                 *self.param_space['colsample_bytree']),
            'gamma': trial.suggest_float('gamma', 
                                      *self.param_space['gamma']),
            'alpha': trial.suggest_float('alpha', 
                                      *self.param_space['alpha']),
            'lambda': trial.suggest_float('lambda', 
                                       *self.param_space['lambda']),
            'n_estimators': trial.suggest_int('n_estimators', 
                                           *self.param_space['n_estimators']),
            'random_state': 42
        }
        
        model = XGBoostRegressor(params)
        results = model.train(X_train, y_train, X_val, y_val, early_stopping_rounds=30, verbose=False)
        
        return results['val_rmse']
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray, 
                X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        
        logger.info(f"Starting Optuna optimization with {self.n_trials} trials")
        
        self.study = optuna.create_study(direction=self.direction, study_name=self.study_name)
        
        self.study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        self.best_params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': 42
        })
        
        logger.info(f"Optimization completed. Best value: {self.study.best_value}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def save_study(self, file_path: str) -> None:
        
        if self.study is None:
            raise ValueError("No study to save. Run optimize() first.")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
        logger.info(f"Saving Optuna study to {file_path}")
        with open(file_path, 'wb') as f:
            pickle.dump(self.study, f)
    
    def load_study(self, file_path: str) -> None:
       
        logger.info(f"Loading Optuna study from {file_path}")
        with open(file_path, 'rb') as f:
            self.study = pickle.load(f)
            
        self.best_params = self.study.best_params
        self.best_params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': 42
        })
    
    def get_best_model(self) -> XGBoostRegressor:
       
        if self.best_params is None:
            raise ValueError("No best parameters available. Run optimize() first.")
            
        return XGBoostRegressor(self.best_params)


if __name__ == "__main__":
    # Testing the Optuna tuner
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Optuna hyperparameter tuning for XGBoost')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data file')
    parser.add_argument('--output_path', type=str, default='./models/', help='Path to save results')
    parser.add_argument('--n_trials', type=int, default=30, help='Number of Optuna trials')
    
    args = parser.parse_args()
    
    try:
        with open(args.data_path, 'rb') as f:
            data = pickle.load(f)
        
        X_train = data.get('X_train')
        y_train = data.get('y_train')
        X_val = data.get('X_val')
        y_val = data.get('y_val')
        
        if X_train is not None and y_train is not None and X_val is not None and y_val is not None:
            tuner = OptunaHyperparameterTuner(n_trials=args.n_trials)
            best_params = tuner.optimize(X_train, y_train, X_val, y_val)
            
            os.makedirs(args.output_path, exist_ok=True)
            study_path = os.path.join(args.output_path, 'optuna_study.pkl')
            tuner.save_study(study_path)
            
            model = tuner.get_best_model()
            model.train(X_train, y_train, X_val, y_val)
            model.save_model(os.path.join(args.output_path, 'best_model.json'))
            
            logger.info(f"Study saved to {study_path}")
            logger.info(f"Best model saved to {os.path.join(args.output_path, 'best_model.json')}")
        else:
            logger.error("Data doesn't contain required training and validation sets")
    except Exception as e:
        logger.error(f"Error in Optuna tuning: {e}")

        