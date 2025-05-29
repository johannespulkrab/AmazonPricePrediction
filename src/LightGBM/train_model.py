import json
import joblib 
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from data_preprocessing import load_or_assemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

MODEL_OUT_PATH  = '/home/mlt_ml1/AmazonPricePrediction/src/LightGBM/lgbm_final_model_1.pkl'

with open('/home/mlt_ml1/AmazonPricePrediction/src/LightGBM/best_params.json') as f:
    best_params = json.load(f)


best_params.update({
    "objective": "regression",
    "metric": "mae",
    "random_state": 42,
    "n_jobs": 2
})

def train_and_save_model():
    X, metadata, y = load_or_assemble(force_rebuild=False)

    # 1) final train/test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # 2) carve out a small validation set from the training block
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42
    )

    # 3) scale
    scl = StandardScaler(with_mean=False)
    X_train = scl.fit_transform(X_train)
    X_val   = scl.transform(X_val)
    X_test  = scl.transform(X_test)

    
    model = LGBMRegressor(**best_params)

  
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=20) ]
    )

    #  final test performance
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Final test MAE: {mae:.5f}")

    
    joblib.dump({'model': model, 'scaler': scl}, MODEL_OUT_PATH)
    print(f"Saved model+scaler to '{MODEL_OUT_PATH}'")

if __name__ == "__main__":
    train_and_save_model()
    print("Done.")
