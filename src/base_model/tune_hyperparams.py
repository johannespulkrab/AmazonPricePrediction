import argparse
import logging
import json
import time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from data_loader import load_processed_data, prepare_data_splits
from model import BaselineRegressor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def tune_ridge_lasso(X_train, y_train, X_val, y_val, model_type='ridge'):
    """Tune alpha parameter for Ridge or Lasso regression"""

    alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    best_alpha = None
    best_score = float('inf')
    results = []

    logger.info(f"Tuning {model_type} regression with {len(alpha_values)} alpha values")

    for alpha in alpha_values:
        logger.info(f"Testing alpha = {alpha}")

        model = BaselineRegressor(model_type=model_type, alpha=alpha)
        training_results = model.train(X_train, y_train, X_val, y_val)

        val_rmse = training_results['val_metrics']['rmse']

        results.append({
            'alpha': alpha,
            'val_rmse': val_rmse,
            'train_rmse': training_results['train_metrics']['rmse'],
            'val_mae': training_results['val_metrics']['mae'],
            'val_r2': training_results['val_metrics']['r2']
        })

        if val_rmse < best_score:
            best_score = val_rmse
            best_alpha = alpha

        logger.info(f"Alpha {alpha}: Validation RMSE = {val_rmse:.4f}")

    logger.info(f"Best alpha for {model_type}: {best_alpha} (RMSE: {best_score:.4f})")

    return {
        'best_alpha': best_alpha,
        'best_rmse': best_score,
        'all_results': results
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for baseline models')

    parser.add_argument('--processed_dir', type=str, default='./data/processed',
                        help='Directory containing processed data files')
    parser.add_argument('--model_types', nargs='+', default=['linear', 'ridge', 'lasso'],
                        choices=['linear', 'ridge', 'lasso'],
                        help='Model types to tune')

    parser.add_argument('--use_subset', action='store_true',
                        help='Use subset of data for faster tuning')
    parser.add_argument('--subset_ratio', type=float, default=0.3,
                        help='Ratio of data to use when using subset')

    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Proportion of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Proportion of data for validation')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Proportion of data for testing')

    parser.add_argument('--output_dir', type=str, default='./outputs/baseline_tuning',
                        help='Directory to save tuning results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(output_dir / 'tuning.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("Starting hyperparameter tuning for baseline models")
    logger.info(f"Arguments: {args}")

    start_time = time.time()

    try:
        logger.info(f"Loading data from {args.processed_dir}")
        X, y = load_processed_data(args.processed_dir)
        logger.info(f"Loaded data with shape: X={X.shape}, y={y.shape}")

        if args.use_subset:
            logger.info(f"Using {args.subset_ratio * 100}% of data for faster tuning")
            X_subset, _, y_subset, _ = train_test_split(
                X, y,
                train_size=args.subset_ratio,
                random_state=args.seed
            )
            X, y = X_subset, y_subset
            logger.info(f"Subset shape: X={X.shape}, y={y.shape}")

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

        all_results = {}

        for model_type in args.model_types:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Tuning {model_type.upper()} model")
            logger.info(f"{'=' * 50}")

            if model_type == 'linear':
                logger.info("Linear regression has no hyperparameters to tune")
                model = BaselineRegressor(model_type='linear')
                training_results = model.train(X_train, y_train, X_val, y_val)
                test_metrics = model.evaluate(X_test, y_test)

                all_results[model_type] = {
                    'val_rmse': training_results['val_metrics']['rmse'],
                    'test_rmse': test_metrics['rmse'],
                    'test_mae': test_metrics['mae'],
                    'test_r2': test_metrics['r2']
                }

            else:
                tuning_results = tune_ridge_lasso(X_train, y_train, X_val, y_val, model_type)

                best_alpha = tuning_results['best_alpha']
                logger.info(f"Training final {model_type} model with best alpha: {best_alpha}")

                final_model = BaselineRegressor(model_type=model_type, alpha=best_alpha)
                final_model.train(X_train, y_train)
                test_metrics = final_model.evaluate(X_test, y_test)

                all_results[model_type] = {
                    'best_alpha': best_alpha,
                    'val_rmse': tuning_results['best_rmse'],
                    'test_rmse': test_metrics['rmse'],
                    'test_mae': test_metrics['mae'],
                    'test_r2': test_metrics['r2'],
                    'tuning_details': tuning_results['all_results']
                }

        tuning_time = time.time() - start_time

        final_results = {
            'model_results': all_results,
            'tuning_time': tuning_time,
            'dataset_info': {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'used_subset': args.use_subset,
                'subset_ratio': args.subset_ratio if args.use_subset else 1.0
            },
            'best_model': min(all_results.keys(),
                              key=lambda k: all_results[k]['test_rmse'])
        }

        results_path = output_dir / 'tuning_results.json'
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"Tuning results saved to {results_path}")

        summary_path = output_dir / 'tuning_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("Baseline Models Hyperparameter Tuning Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Tuning Time: {tuning_time:.2f} seconds\n")
            f.write(f"Dataset: {X.shape[0]:,} samples, {X.shape[1]:,} features\n")
            if args.use_subset:
                f.write(f"Used subset: {args.subset_ratio * 100}% of original data\n")
            f.write("\nResults by Model:\n")
            f.write("-" * 30 + "\n")

            for model_type, results in all_results.items():
                f.write(f"\n{model_type.upper()}:\n")
                if 'best_alpha' in results:
                    f.write(f"  Best Alpha: {results['best_alpha']}\n")
                f.write(f"  Validation RMSE: {results['val_rmse']:.4f}\n")
                f.write(f"  Test RMSE: {results['test_rmse']:.4f}\n")
                f.write(f"  Test MAE: {results['test_mae']:.4f}\n")
                f.write(f"  Test R²: {results['test_r2']:.4f}\n")

            best_model = final_results['best_model']
            f.write(f"\nBest Model: {best_model.upper()}\n")
            f.write(f"Best Test RMSE: {all_results[best_model]['test_rmse']:.4f}\n")

        logger.info(f"Summary saved to {summary_path}")

        logger.info("\n" + "=" * 50)
        logger.info("Hyperparameter tuning completed")
        logger.info(f"Best model: {final_results['best_model'].upper()}")
        logger.info(f"Best test RMSE: {all_results[final_results['best_model']]['test_rmse']:.4f}")
        logger.info(f"Total tuning time: {tuning_time:.2f} seconds")
        logger.info("=" * 50)

        print("\nHyperparameter Tuning Results:")
        for model_type, results in all_results.items():
            print(f"  {model_type.upper()}: Test RMSE = {results['test_rmse']:.4f}")
        print(f"  Best Model: {final_results['best_model'].upper()}")

    except Exception as e:
        logger.error(f"Error during tuning: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()