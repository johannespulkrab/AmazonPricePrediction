import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import json
from tqdm import tqdm
from data_loader import load_and_prepare_data

# Load data
X, y = load_and_prepare_data(force_rebuild=False)

# Use only 30% of the dataset for much faster hyperparameter tuning
X_subset, _, y_subset, _ = train_test_split(X, y, test_size=0.7, random_state=42)
print(f"Using subset: {X_subset.shape[0]} samples (30% of original {X.shape[0]})")

# Split subset: 75% for training, 25% for final test
X_train_val, X_test, y_train_val, y_test = train_test_split(X_subset, y_subset, test_size=0.25, random_state=42)

# Split training data: 75% train, 25% validation (for hyperparameter tuning)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

PARAMS_OUT = 'best_params_xgb.json'

# Progress tracking
pbar = None
trial_count = 0

def objective(trial):
    global pbar, trial_count
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 8), 
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0), 
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0), 
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'random_state': 42,
        'n_jobs': -1 
    }

    model = xgb.XGBRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    
    trial_count += 1
    if pbar is not None:
        try:
            best_mae = f'{trial.study.best_value:.3f}' if trial.study.best_value is not None else 'N/A'
        except (ValueError, AttributeError):
            best_mae = 'N/A'
            
        pbar.set_postfix({
            'Trial': trial_count, 
            'Current MAE': f'{mae:.3f}',
            'Best MAE': best_mae
        })
        pbar.update(1)
    
    return mae

if __name__ == "__main__":
    print("Starting optimized XGBoost hyperparameter optimization...")
    print(f"Data shapes: X_train={X_train.shape}, X_val={X_val.shape}")
    
    n_trials = 20 
    
    pbar = tqdm(total=n_trials, desc="Optimizing XGBoost", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]')
    
    study = optuna.create_study(direction='minimize')
    
    # Add some resilience to the optimization
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
    except Exception as e:
        print(f"Optimization encountered an error: {e}")
        print("Continuing with completed trials...")
    
    # Check if we have any successful trials
    if len(study.trials) == 0 or study.best_value is None:
        print("No successful trials completed!")
        exit(1)
    
    # Close progress bar
    pbar.close()
    
    # Get best parameters
    best_params = study.best_params
    best_params.update({
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'random_state': 42,
        'n_jobs': -1  # Use all cores for final model
    })
    
    print(f"\nOptimization completed!")
    print(f"Best MAE: {study.best_value:.5f}")
    print("Best hyperparameters:", study.best_params)
    
    # Save best parameters
    with open(PARAMS_OUT, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"Best parameters saved to {PARAMS_OUT}")