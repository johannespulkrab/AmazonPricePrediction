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
        self.param_space = param_space or self.default_param_space()
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
        
        #print("Optimization completed. Best value: {}".format(self.study.best_value))
        #print('Best paramet