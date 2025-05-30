import argparse
import logging
import pickle
import json
import time
from pathlib import Path

from data_loader import load_processed_data, prepare_data_splits
from model import BaselineRegressor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Amazon Price Prediction Baseline Model')

    parser.add_argument('--processed_dir', type=str, default='./data/processed',
                        help='Directory containing processed data files')
    parser.add_argument('--model_type', type=str, default='linear',
                        choices=['linear', 'ridge', 'lasso'],
                        help='Type of linear model to use')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Regularization strength for Ridge/Lasso')

    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Proportion of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Proportion of data for validation')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Proportion of data for testing')

    parser.add_argument('--output_dir', type=str, default='./outputs/baseline_model',
                        help='Directory to save outputs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(output_dir / 'training.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("Starting Amazon Price Prediction Baseline Training")
    logger.info(f"Arguments: {args}")

    start_time = time.time()

    try:
        logger.info(f"Loading data from {args.processed_dir}")
        X, y = load_processed_data(args.processed_dir)
        logger.info(f"Loaded data with shape: X={X.shape}, y={y.shape}")

        logger.info("Preparing data splits...")
        data_splits = prepare_data_splits(
            X, y,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.seed
        )

        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        X_val = data_splits['X_val']
        y_val = data_splits['y_val']
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        scaler = data_splits['scaler']

        splits_path = output_dir / 'data_splits.pkl'
        with open(splits_path, 'wb') as f:
            pickle.dump(data_splits, f)
        logger.info(f"Data splits saved to {splits_path}")

        logger.info(f"Initializing {args.model_type} regression model")
        model = BaselineRegressor(model_type=args.model_type, alpha=args.alpha)

        logger.info("Training model...")
        training_results = model.train(X_train, y_train, X_val, y_val)

        model_path = output_dir / f'{args.model_type}_model.pkl'
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

        scaler_path = output_dir / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved to {scaler_path}")

        logger.info("Evaluating on test set...")
        test_metrics = model.evaluate(X_test, y_test)

        training_time = time.time() - start_time

        results = {
            'model_type': args.model_type,
            'alpha': args.alpha,
            'training_results': training_results,
            'test_metrics': test_metrics,
            'training_time': training_time,
            'data_shape': {
                'n_samples': X.shape[0],
                'n_features': X.shape[1]
            }
        }

        results_path = output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

        results_pkl_path = output_dir / 'results.pkl'
        with open(results_pkl_path, 'wb') as f:
            pickle.dump(results, f)

        report_path = output_dir / 'results_summary.txt'
        with open(report_path, 'w') as f:
            f.write("Amazon Price Prediction - Baseline Model Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Type: {args.model_type.upper()}\n")
            if args.model_type in ['ridge', 'lasso']:
                f.write(f"Alpha (regularization): {args.alpha}\n")
            f.write(f"Training Time: {training_time:.2f} seconds\n")
            f.write(f"Dataset Size: {X.shape[0]:,} samples, {X.shape[1]:,} features\n\n")

            f.write("Training Metrics:\n")
            train_metrics = training_results['train_metrics']
            for metric, value in train_metrics.items():
                f.write(f"  {metric.upper()}: {value:.4f}\n")

            if 'val_metrics' in training_results:
                f.write("\nValidation Metrics:\n")
                val_metrics = training_results['val_metrics']
                for metric, value in val_metrics.items():
                    f.write(f"  {metric.upper()}: {value:.4f}\n")

            f.write("\nTest Metrics:\n")
            for metric, value in test_metrics.items():
                f.write(f"  {metric.upper()}: {value:.4f}\n")

        logger.info(f"Results summary saved to {report_path}")

        logger.info("=" * 50)
        logger.info("Baseline training completed successfully")
        logger.info(f"Model Type: {args.model_type}")
        logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"Test MAE: {test_metrics['mae']:.4f}")
        logger.info(f"Test R²: {test_metrics['r2']:.4f}")
        logger.info(f"Training time: {training_time:.2f} seconds")
        logger.info("=" * 50)

        print("\nBaseline Model Results:")
        print(f"  Model Type: {args.model_type}")
        print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"  Test MAE: {test_metrics['mae']:.4f}")
        print(f"  Test R²: {test_metrics['r2']:.4f}")
        print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Model saved to: {model_path}")

    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()