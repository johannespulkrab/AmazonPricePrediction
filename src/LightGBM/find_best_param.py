import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor, early_stopping

from data_preprocessing import load_or_assemble

import json



def train_and_save_model():
    X, metadata, y = load_or_assemble(force_rebuild=False)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)
PARAMS_OUT = 'best_params.json'

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_data_in_leaf', 10, 50),
        'colsample_bytree': trial.suggest_uniform('feature_fraction', 0.7, 1.0),
        "force_col_wise": True,
        'random_state': 42,
        "n_jobs": 2,
    }

    model = LGBMRegressor(**param)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric= "mae", callbacks=[early_stopping(stopping_rounds=100)])
    preds = model.predict(X_val)
    rmae = mean_absolute_error(y_val, preds)
    return rmae



study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, n_jobs = 4)
best_params = study.best_params

print(f"Best RMAE: {study.best_value:.5f}")  #33.73714
print("Best hyperparameters:", study.best_params)

with open(PARAMS_OUT, 'w') as f:
    json.dump(best_params, f, indent=2)