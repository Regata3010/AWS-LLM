# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from ..src.logger import logging
# from ..src.exception import CustomException
# from core.bias_detector.preprocessor import preprocess_data
# import streamlit as st
# import sys

# def get_regression_model(choice):
#     if choice == "Linear Regression":
#         return LinearRegression()
#     elif choice == "Random Forest Regressor":
#         return RandomForestRegressor()
#     elif choice == "XGBoost Regressor":
#         return XGBRegressor(objective='reg:squarederror')
#     else:
#         raise CustomException(f"Unsupported model choice: {choice}", sys)


# def train_regression_model(X_train,X_test,y_train,y_test,model_choice):
#     try:
#         model = get_regression_model(model_choice)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         mse = mean_squared_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)

#         st.write(f"✅ MSE: {mse}")
#         st.write(f"✅ R² Score: {r2}")

#         return model, y_pred
#     except Exception as e:
#         raise CustomException(e,sys)
#--------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import sys
import os

from core.src.logger import logging
from core.src.exception import CustomException


@dataclass 
class RegressionModelConfig:
    """Configuration for regression model training"""
    name: str
    model_class: Any
    default_params: Dict
    param_grid: Optional[Dict] = None
    requires_scaling: bool = False
    bias_aware: bool = False


@dataclass
class RegressionTrainingResult:
    """Standardized regression training result structure"""
    model_id: str
    model_type: str
    model_name: str
    performance_metrics: Dict
    cross_val_scores: Dict
    training_time: float
    model_path: str
    hyperparams: Dict
    feature_importance: Optional[Dict] = None
    model_object: Any = None
    bias_aware: bool = False


class EnterpriseRegressionTrainer:
    """Enterprise-grade regression model training service"""
    
    def __init__(self, models_storage_path: str = "models/"):
        self.models_storage_path = models_storage_path
        os.makedirs(models_storage_path, exist_ok=True)
        
        # Define available regression models with their configurations
        self.model_configs = {
            "linear_regression": RegressionModelConfig(
                name="Linear Regression",
                model_class=LinearRegression,
                default_params={},
                param_grid=None,  # Linear regression has no hyperparameters to tune
                requires_scaling=True
            ),
            
            "ridge_regression": RegressionModelConfig(
                name="Ridge Regression",
                model_class=Ridge,
                default_params={"random_state": 42},
                param_grid={
                    "alpha": [0.1, 1.0, 10.0, 100.0],
                    "solver": ["auto", "svd", "cholesky", "lsqr"]
                },
                requires_scaling=True
            ),
            
            "lasso_regression": RegressionModelConfig(
                name="Lasso Regression", 
                model_class=Lasso,
                default_params={"random_state": 42, "max_iter": 2000},
                param_grid={
                    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                    "selection": ["cyclic", "random"]
                },
                requires_scaling=True
            ),
            
            "random_forest": RegressionModelConfig(
                name="Random Forest Regressor",
                model_class=RandomForestRegressor,
                default_params={"random_state": 42, "n_jobs": -1},
                param_grid={
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None]
                }
            ),
            
            "gradient_boosting": RegressionModelConfig(
                name="Gradient Boosting Regressor",
                model_class=GradientBoostingRegressor,
                default_params={"random_state": 42},
                param_grid={
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 0.9, 1.0]
                }
            ),
            
            "xgboost": RegressionModelConfig(
                name="XGBoost Regressor",
                model_class=XGBRegressor,
                default_params={
                    "random_state": 42,
                    "objective": "reg:squarederror"
                },
                param_grid={
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "subsample": [0.8, 0.9, 1.0],
                    "colsample_bytree": [0.8, 0.9, 1.0]
                }
            )
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available regression model types"""
        return list(self.model_configs.keys())
    
    def _create_model(self, model_type: str, hyperparams: Optional[Dict] = None) -> Any:
        """Create regression model instance with specified parameters"""
        try:
            config = self.model_configs[model_type]
            params = config.default_params.copy()
            
            if hyperparams:
                params.update(hyperparams)
                
            model = config.model_class(**params)
            logging.info(f"Created {config.name} with params: {params}")
            return model
            
        except KeyError:
            raise CustomException(f"Unsupported regression model type: {model_type}", sys)
        except Exception as e:
            raise CustomException(f"Error creating regression model {model_type}: {str(e)}", sys)
    
    def _calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive regression performance metrics"""
        try:
            metrics = {
                "mse": mean_squared_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "mae": mean_absolute_error(y_true, y_pred),
                "r2_score": r2_score(y_true, y_pred)
            }
            
            # Add MAPE if no zero values in y_true (to avoid division by zero)
            if not np.any(y_true == 0):
                metrics["mape"] = mean_absolute_percentage_error(y_true, y_pred)
            
            # Add explained variance
            from sklearn.metrics import explained_variance_score
            metrics["explained_variance"] = explained_variance_score(y_true, y_pred)
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating regression metrics: {str(e)}")
            return {"error": str(e)}
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict]:
        """Extract feature importance if available"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                # Sort by importance
                sorted_importance = dict(sorted(importance_dict.items(), 
                                              key=lambda x: x[1], reverse=True))
                return sorted_importance
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute values of coefficients
                coef_dict = dict(zip(feature_names, np.abs(model.coef_)))
                sorted_coef = dict(sorted(coef_dict.items(), 
                                        key=lambda x: x[1], reverse=True))
                return sorted_coef
            else:
                return None
                
        except Exception as e:
            logging.warning(f"Could not extract feature importance: {str(e)}")
            return None
    
    def _save_model(self, model: Any, model_id: str) -> str:
        """Save trained regression model to disk"""
        try:
            model_filename = f"{model_id}.joblib"
            model_path = os.path.join(self.models_storage_path, model_filename)
            joblib.dump(model, model_path)
            logging.info(f"Regression model saved to: {model_path}")
            return model_path
            
        except Exception as e:
            logging.error(f"Error saving regression model: {str(e)}")
            raise CustomException(f"Failed to save regression model: {str(e)}", sys)
    
    def train_single_model(self, 
                          X_train: pd.DataFrame, 
                          X_test: pd.DataFrame,
                          y_train: np.ndarray, 
                          y_test: np.ndarray,
                          model_type: str,
                          optimize_hyperparams: bool = False,
                          cv_folds: int = 5) -> RegressionTrainingResult:
        """Train a single regression model with comprehensive evaluation"""
        
        try:
            logging.info(f"Starting regression training for {model_type}")
            start_time = time.time()
            model_id = str(uuid.uuid4())
            
            config = self.model_configs[model_type]
            
            # Scale features if required
            X_train_processed = X_train.copy()
            X_test_processed = X_test.copy()
            scaler = None
            
            if config.requires_scaling:
                scaler = StandardScaler()
                X_train_processed = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index
                )
                X_test_processed = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )
                logging.info(f"Applied feature scaling for {model_type}")
            
            # Hyperparameter optimization if requested
            best_params = config.default_params.copy()
            if optimize_hyperparams and config.param_grid:
                logging.info(f"Starting hyperparameter optimization for {model_type}")
                
                base_model = self._create_model(model_type)
                grid_search = GridSearchCV(
                    base_model, 
                    config.param_grid,
                    cv=cv_folds,
                    scoring='neg_mean_squared_error',  # Use MSE for regression
                    n_jobs=-1
                )
                
                grid_search.fit(X_train_processed, y_train)
                best_params.update(grid_search.best_params_)
                model = grid_search.best_estimator_
                
                logging.info(f"Best params for {model_type}: {grid_search.best_params_}")
            else:
                # Train with default parameters
                model = self._create_model(model_type, best_params)
                model.fit(X_train_processed, y_train)
            
            # Generate predictions
            y_pred = model.predict(X_test_processed)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(y_test, y_pred)
            
            # Cross-validation scores
            cv_scores = {}
            try:
                cv_r2 = cross_val_score(model, X_train_processed, y_train, cv=cv_folds, scoring='r2')
                cv_mse = cross_val_score(model, X_train_processed, y_train, cv=cv_folds, 
                                       scoring='neg_mean_squared_error')
                cv_mae = cross_val_score(model, X_train_processed, y_train, cv=cv_folds, 
                                       scoring='neg_mean_absolute_error')
                
                cv_scores = {
                    "cv_r2_mean": cv_r2.mean(),
                    "cv_r2_std": cv_r2.std(),
                    "cv_mse_mean": -cv_mse.mean(),  # Convert back to positive MSE
                    "cv_mse_std": cv_mse.std(),
                    "cv_mae_mean": -cv_mae.mean(),  # Convert back to positive MAE
                    "cv_mae_std": cv_mae.std()
                }
            except Exception as e:
                logging.warning(f"Cross-validation failed: {str(e)}")
                cv_scores = {"error": "Cross-validation failed"}
            
            # Feature importance
            feature_importance = self._get_feature_importance(model, X_train.columns.tolist())
            
            # Save model
            model_path = self._save_model(model, model_id)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Create result object
            result = RegressionTrainingResult(
                model_id=model_id,
                model_type=model_type,
                model_name=config.name,
                performance_metrics=performance_metrics,
                cross_val_scores=cv_scores,
                training_time=training_time,
                model_path=model_path,
                hyperparams=best_params,
                feature_importance=feature_importance,
                model_object=model,
                bias_aware=config.bias_aware
            )
            
            # logging.info(f"✅ Regression training completed for {model_type} in {training_time:.2f}s")
            # logging.info(f"📊 R² Score: {performance_metrics.get('r2_score', 'N/A'):.4f}")
            # logging.info(f"📊 RMSE: {performance_metrics.get('rmse', 'N/A'):.4f}")
            
            return result
            
        except Exception as e:
            logging.error(f"Regression training failed for {model_type}: {str(e)}")
            raise CustomException(f"Regression training failed for {model_type}: {str(e)}", sys)
    
    def train_multiple_models(self,
                            X_train: pd.DataFrame,
                            X_test: pd.DataFrame, 
                            y_train: np.ndarray,
                            y_test: np.ndarray,
                            model_types: Optional[List[str]] = None,
                            optimize_hyperparams: bool = False,
                            cv_folds: int = 5) -> List[RegressionTrainingResult]:
        """Train multiple regression models and return comparison results"""
        
        try:
            if model_types is None:
                model_types = list(self.model_configs.keys())
            
            logging.info(f"Training {len(model_types)} regression models: {model_types}")
            
            results = []
            for model_type in model_types:
                try:
                    result = self.train_single_model(
                        X_train, X_test, y_train, y_test, 
                        model_type, optimize_hyperparams, cv_folds
                    )
                    results.append(result)
                    
                except Exception as e:
                    logging.error(f"Failed to train regression model {model_type}: {str(e)}")
                    continue
            
            # Sort by R² score (best first)
            results.sort(key=lambda x: x.performance_metrics.get('r2_score', 0), reverse=True)
            
            logging.info(f"✅ Successfully trained {len(results)} regression models")
            return results
            
        except Exception as e:
            logging.error(f"Multi-model regression training failed: {str(e)}")
            raise CustomException(f"Multi-model regression training failed: {str(e)}", sys)
    
    def load_model(self, model_path: str) -> Any:
        """Load a saved regression model"""
        try:
            model = joblib.load(model_path)
            logging.info(f"Regression model loaded from: {model_path}")
            return model
        except Exception as e:
            raise CustomException(f"Failed to load regression model from {model_path}: {str(e)}", sys)
    
    def get_model_summary(self, results: List[RegressionTrainingResult]) -> Dict:
        """Generate summary of multiple regression model training results"""
        if not results:
            return {"error": "No training results provided"}
        
        summary = {
            "total_models": len(results),
            "best_model": {
                "name": results[0].model_name,
                "type": results[0].model_type,
                "r2_score": results[0].performance_metrics.get('r2_score', 0),
                "rmse": results[0].performance_metrics.get('rmse', 0),
                "mae": results[0].performance_metrics.get('mae', 0)
            },
            "model_rankings": []
        }
        
        for i, result in enumerate(results, 1):
            summary["model_rankings"].append({
                "rank": i,
                "model_name": result.model_name,
                "model_type": result.model_type,
                "r2_score": result.performance_metrics.get('r2_score', 0),
                "rmse": result.performance_metrics.get('rmse', 0),
                "mae": result.performance_metrics.get('mae', 0),
                "training_time": result.training_time
            })
        
        return summary


# Convenience functions for backward compatibility and ease of use
def get_regression_model(choice: str) -> Any:
    """Get a regression model instance (backward compatibility)"""
    trainer = EnterpriseRegressionTrainer()
    model_mapping = {
        "Linear Regression": "linear_regression",
        "Random Forest Regressor": "random_forest",
        "XGBoost Regressor": "xgboost",
        "Ridge Regression": "ridge_regression",
        "Lasso Regression": "lasso_regression"
    }
    
    model_type = model_mapping.get(choice)
    if not model_type:
        raise CustomException(f"Unsupported regression model choice: {choice}", sys)
    
    return trainer._create_model(model_type)


def train_regression_model(X_train: pd.DataFrame,
                         X_test: pd.DataFrame, 
                         y_train: np.ndarray,
                         y_test: np.ndarray,
                         model_choice: str,
                         optimize_hyperparams: bool = False) -> Tuple[Any, np.ndarray, RegressionTrainingResult]:
    """Train a single regression model (backward compatibility)"""
    
    trainer = EnterpriseRegressionTrainer()
    
    # Map old choice names to new model types
    model_mapping = {
        "Linear Regression": "linear_regression",
        "Random Forest Regressor": "random_forest", 
        "XGBoost Regressor": "xgboost",
        "Ridge Regression": "ridge_regression",
        "Lasso Regression": "lasso_regression"
    }
    
    model_type = model_mapping.get(model_choice)
    if not model_type:
        raise CustomException(f"Unsupported regression model choice: {model_choice}", sys)
    
    result = trainer.train_single_model(
        X_train, X_test, y_train, y_test, 
        model_type, optimize_hyperparams
    )
    
    # Generate predictions for backward compatibility
    y_pred = result.model_object.predict(X_test)
    
    return result.model_object, y_pred, result


# if __name__ == "__main__":
#     # Example usage and testing
#     from sklearn.datasets import make_regression
    
#     # Generate sample regression data
#     X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
#     X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Test enterprise regression trainer
#     trainer = EnterpriseRegressionTrainer()
    
#     # Train single model
#     print("Testing single regression model training...")
#     result = trainer.train_single_model(X_train, X_test, y_train, y_test, "random_forest")
#     print(f"✅ Single model result: {result.model_name} - R²: {result.performance_metrics.get('r2_score'):.4f}")
    
#     # Train multiple models
#     print("\nTesting multiple regression model training...")
#     results = trainer.train_multiple_models(X_train, X_test, y_train, y_test)
    
#     summary = trainer.get_model_summary(results)
#     print(f"✅ Best regression model: {summary['best_model']['name']} - R²: {summary['best_model']['r2_score']:.4f}")