import json
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from data_loader import load_and_prepare_data

MODEL_OUT_PATH = 'xgb_final_model.pkl'
BEST_PARAMS_PATH = '/home/mlt_ml1/AmazonPricePrediction/src/XGBoost/best_params_xgb.json'

def load_best_params(params_path):
    """Load the best hyperparameters from Optuna optimization"""
    try:
        with open(params_path, 'r') as f:
            best_params = json.load(f)
        print("Best hyperparameters loaded successfully:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        return best_params
    except FileNotFoundError:
        print(f"Error: {params_path} not found. Run hyperparameter optimization first!")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {params_path}")
        return None

def train_and_save_model():
    """Train final XGBoost model with optimized hyperparameters"""
    
    # Load best hyperparameters
    best_params = load_best_params(BEST_PARAMS_PATH)
    if best_params is None:
        return
    
    print("\n" + "="*50)
    print("TRAINING FINAL XGBOOST MODEL")
    print("="*50)
    
    # Load FULL dataset
    print("Loading full dataset...")
    X, y = load_and_prepare_data(force_rebuild=False)
    print(f"Full dataset shape: {X.shape}")
    
    # Split data: 75% train, 25% test (like your colleague's approach)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Further split training: 90% train, 10% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42
    )
    
    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
    
    print("\nApplying scaling...")
    scaler = StandardScaler(with_mean=False)  
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    model_params = best_params.copy()
    
    # Remove any Optuna-specific or problematic parameters
    params_to_remove = ['early_stopping_rounds']
    for param in params_to_remove:
        model_params.pop(param, None)
    
    # Ensure proper parameter format
    model_params['random_state'] = 42
    model_params['n_jobs'] = -1  # Use all cores
    
    print("Final model parameters:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    
    # Create and train model
    model = xgb.XGBRegressor(**model_params)
    
    print("\nTraining model...")
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=True  # Show training progress
    )
    
    # Evaluate model
    print("\n" + "="*30)
    print("MODEL EVALUATION")
    print("="*30)
    
    # Validation performance
    val_preds = model.predict(X_val_scaled)
    val_mae = mean_absolute_error(y_val, val_preds)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    
    # Test performance (final evaluation)
    test_preds = model.predict(X_test_scaled)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    
    print(f"Validation MAE: {val_mae:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # Feature importance (top 10)
    print(f"\nTop 10 Feature Importances:")
    feature_importance = model.feature_importances_
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
    
    # Save model and scaler together
    print(f"\nSaving model and scaler to {MODEL_OUT_PATH}...")
    model_package = {
        'model': model,
        'scaler': scaler,
        'best_params': best_params,
        'performance': {
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'test_mae': test_mae,
            'test_rmse': test_rmse
        }
    }
    
    joblib.dump(model_package, MODEL_OUT_PATH)
    print(f"Model saved successfully!")
    
    return model, scaler, test_mae

def load_and_predict_example():
    """Example of how to load the saved model and make predictions"""
    print("\n" + "="*40)
    print("LOADING MODEL FOR PREDICTION EXAMPLE")
    print("="*40)
    
    try:
        model_package = joblib.load(MODEL_OUT_PATH)
        model = model_package['model']
        scaler = model_package['scaler']
        performance = model_package['performance']
        
        print("Model loaded successfully!")
        print(f"Test MAE: {performance['test_mae']:.4f}")
        
        
        return model, scaler
        
    except FileNotFoundError:
        print(f"Model file {MODEL_OUT_PATH} not found. Train the model first!")
        return None, None

if __name__ == "__main__":
    print("XGBoost Final Training Script")
    print("="*50)
    
    # Train final model
    model, scaler, test_mae = train_and_save_model()
    
    if model is not None:
        print(f"\Training completed successfully!")
        print(f"Final Test MAE: {test_mae:.4f}")
        
        load_and_predict_example()
    else:
        print("Training failed!")