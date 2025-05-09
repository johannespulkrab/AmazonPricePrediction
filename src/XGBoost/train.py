''' Main training script'''

import os
import argparse
import logging
import pickle
import numpy as np
import time
from typing import Any, Dict
from tqdm import tqdm

from src.XGBoost.data_loader import load_and_prepare_data
from src.XGBoost.model import XGBoostRegressor
from src.XGBoost.tuner import OptunaHyperparameterTuner


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Amazon Price Prediction with XGBoost')
    
    parser.add_argument('--processed_dir', type=str, required=True,
                       help='Directory containing processed data files')
    parser.add_argument('--max_batches', type=int, default=None,
                       help='Maximum number of text feature batches to load (None for all)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Proportion of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Proportion of data for validation')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Proportion of data for testing')
    
    parser.add_argument('--tune_hyperparams', action='store_true',
                       help='Whether to perform hyperparameter tuning')
    parser.add_argument('--subset_for_tuning', action='store_true',
                       help='Use only half of the data for hyperparameter tuning (faster)')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of trials for hyperparameter tuning')
    parser.add_argument('--early_stopping_rounds', type=int, default=50,
                       help='Number of rounds for early stopping')
    
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Directory to save outputs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'training.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("Starting Amazon Price Prediction training")
    logger.info(f"Arguments: {args}")
    
    start_time = time.time()
    
    steps = ["Data Loading", "Hyperparameter Tuning", "Model Training", "Evaluation"]
    
    try:
        with tqdm(total=len(steps), desc="Training Pipeline", position=0) as pbar:
            pbar.set_description(f"Step 1/{len(steps)}: {steps[0]}")
            logger.info(f"Loading data from {args.processed_dir}")
            
            data = load_and_prepare_data(
                args.processed_dir,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                random_state=args.seed,
                max_batches=args.max_batches
            )
            
            X_train, y_train = data['X_train'], data['y_train']
            X_val, y_val = data['X_val'], data['y_val']
            X_test, y_test = data['X_test'], data['y_test']
            
            logger.info(f"Data loaded successfully. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            
            # Save the splits for future use
            splits_path = os.path.join(args.output_dir, 'data_splits.pkl')
            with open(splits_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Data splits saved to {splits_path}")
            
            pbar.update(1)
            
            # Step 2: Hyperparameter tuning
            pbar.set_description(f"Step 2/{len(steps)}: {steps[1]}")
            model_params = None
            if args.tune_hyperparams:
                logger.info("Running hyperparameter tuning with Optuna")
                
                if args.subset_for_tuning:
                    logger.info("Using subset of data for faster hyperparameter tuning")
                    
                    subset_size = len(X_train) // 2
                    rng = np.random.RandomState(args.seed)
                    subset_indices = rng.choice(len(X_train), subset_size, replace=False)
                    X_train_subset = X_train[subset_indices]
                    y_train_subset = y_train[subset_indices]
                    
                    subset_val_size = len(X_val) // 2
                    subset_val_indices = rng.choice(len(X_val), subset_val_size, replace=False)
                    X_val_subset = X_val[subset_val_indices]
                    y_val_subset = y_val[subset_val_indices]
                    
                    logger.info(f"Tuning on subset - Train: {X_train_subset.shape}, Val: {X_val_subset.shape}")
                    
                    # Use subset for tuning
                    tuner = OptunaHyperparameterTuner(n_trials=args.n_trials)
                    model_params = tuner.optimize(X_train_subset, y_train_subset, X_val_subset, y_val_subset)
                else:
                    tuner = OptunaHyperparameterTuner(n_trials=args.n_trials)
                    model_params = tuner.optimize(X_train, y_train, X_val, y_val)
                
                study_path = os.path.join(args.output_dir, 'optuna_study.pkl')
                tuner.save_study(study_path)
                logger.info(f"Optuna study saved to {study_path}")
                
                import json
                params_path = os.path.join(args.output_dir, 'best_params.json')
                with open(params_path, 'w') as f:
                    json.dump(model_params, f, indent=2)
                logger.info(f"Best parameters saved to {params_path}")
            else:
                logger.info("Skipping hyperparameter tuning, using default parameters")
            
            pbar.update(1)
            
            # Step 3: Create and train model
            pbar.set_description(f"Step 3/{len(steps)}: {steps[2]}")
            logger.info("Training final model")
            
            model = XGBoostRegressor(params=model_params)
            training_results = model.train(
                X_train, y_train, 
                X_val, y_val,
                early_stopping_rounds=args.early_stopping_rounds
            )
            
            model_path = os.path.join(args.output_dir, 'xgboost_model.json')
            model.save_model(model_path)
            logger.info(f"Trained model saved to {model_path}")
            
            pbar.update(1)
            
            # Step 4: Evaluate on test set
            pbar.set_description(f"Step 4/{len(steps)}: {steps[3]}")
            logger.info("Evaluating model on test set")
            
            test_metrics = model.evaluate(X_test, y_test)
            logger.info(f"Test set evaluation: {test_metrics}")
            
            results = {
                'training_results': training_results,
                'test_metrics': test_metrics,
                'parameters': model.params,
                'training_time': time.time() - start_time
            }
            
            results_path = os.path.join(args.output_dir, 'training_results.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            
            report_path = os.path.join(args.output_dir, 'results_summary.txt')
            with open(report_path, 'w') as f:
                f.write("Amazon Price Prediction Results\n")
                f.write("==============================\n\n")
                f.write(f"Training time: {time.time() - start_time:.2f} seconds\n\n")
                f.write("Test Metrics:\n")
                for metric, value in test_metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\nTraining Results:\n")
                for key, value in training_results.items():
                    f.write(f"  {key}: {value}\n")
            
            logger.info(f"Training results saved to {results_path}")
            logger.info(f"Results summary saved to {report_path}")
            
            pbar.update(1)
        
        total_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info("Training completed successfully")
        logger.info(f"Best validation RMSE: {training_results.get('val_rmse')}")
        logger.info(f"Test RMSE: {test_metrics.get('rmse')}")
        logger.info(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        logger.info("=" * 50)
        
        print("\nTraining Results Summary:")
        print(f"  Test RMSE: {test_metrics.get('rmse'):.4f}")
        print(f"  Test MAE: {test_metrics.get('mae'):.4f}")
        print(f"  Test R²: {test_metrics.get('r2'):.4f}")
        print(f"  Training time: {total_time/60:.2f} minutes")
        print(f"  Model saved to: {model_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()


    