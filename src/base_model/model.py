import numpy as np
import pickle
import logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineRegressor:
    def __init__(self, model_type='linear', alpha=1.0):
        """
        Initialize baseline regression model

        Args:
            model_type: 'linear', 'ridge', or 'lasso'
            alpha: regularization strength for ridge/lasso
        """
        self.model_type = model_type
        self.alpha = alpha

        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha, max_iter=2000)
        else:
            raise ValueError("model_type must be 'linear', 'ridge', or 'lasso'")

        self.is_trained = False
        logger.info(f"Initialized {model_type} regression model")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the baseline model"""
        logger.info("Starting model training...")

        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_pred = self.model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_pred)

        results = {
            'train_metrics': train_metrics
        }

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_pred)
            results['val_metrics'] = val_metrics

            logger.info(f"Training RMSE: {train_metrics['rmse']:.4f}")
            logger.info(f"Validation RMSE: {val_metrics['rmse']:.4f}")
        else:
            logger.info(f"Training RMSE: {train_metrics['rmse']:.4f}")

        logger.info("Training completed")
        return results

    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def evaluate(self, X, y_true):
        """Evaluate model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        y_pred = self.predict(X)
        metrics = self._calculate_metrics(y_true, y_pred)

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'alpha': self.alpha
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.alpha = model_data['alpha']
        self.is_trained = True

        logger.info(f"Model loaded from {filepath}")

    def get_feature_importance(self, feature_names=None, top_n=20):
        """Get feature coefficients (importance) for linear models"""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")

        if hasattr(self.model, 'coef_'):
            coefficients = self.model.coef_

            if feature_names is not None:
                importance_data = list(zip(feature_names, coefficients))
                importance_data.sort(key=lambda x: abs(x[1]), reverse=True)
                return importance_data[:top_n]
            else:
                indices = np.argsort(np.abs(coefficients))[::-1]
                return [(f"feature_{i}", coefficients[i]) for i in indices[:top_n]]
        else:
            logger.warning("Model does not have coefficients")
            return None


if __name__ == "__main__":
    logger.info("Baseline model module loaded successfully")