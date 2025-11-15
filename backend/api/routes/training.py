from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import uuid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from configurations.settings import settings
from api.models.requests import TrainingRequest, DetectTaskTypeRequests
from api.models.responses import TrainingResponse, DetectTaskTypeResponse
from api.routes.upload import get_file_from_s3 
from core.src.logger import logging
from core.bias_detector.preprocessor import preprocess_data 
from core.validation.classproblem import type_of_target  
import pickle
import redis
import lightgbm  
import mlflow
from contextlib import contextmanager
from api.models.database import get_db,init_db
from api.models import schemas, crud
from sqlalchemy.orm import Session
import mlflow.sklearn
import tempfile
import os
import time
import json
from mlflow.tracking import MlflowClient
import asyncio
from fastapi.responses import StreamingResponse
from core.validation.data_validator import DataValidator
from deprecated import deprecated

@contextmanager
def safe_training_run(run_name: str):
    """Ensures clean MLflow run management for training"""
    if mlflow.active_run():
        mlflow.end_run()
    
    run = mlflow.start_run(run_name=run_name)
    try:
        yield run
    finally:
        time.sleep(0.2)
        mlflow.end_run()

router = APIRouter()    

# Model storage
model_storage = {}
training_progress: Dict[str, Dict] = {}

BEST_MODELS = {
    "binary": ["XGBoost", "Random Forest", "Logistic Regression", "SVM", "Decision Tree"],
    "multiclass": ["XGBoost", "Random Forest", "Decision Tree", "SVM", "Logistic Regression"],
    "continuous": ["XGBoost Regressor", "Random Forest Regressor", "Linear Regression", "LightGBM Regressor"]
}

MODEL_VALIDATION = {
    # classification-capable
    "XGBoost": ["binary", "multiclass"],
    "Random Forest": ["binary", "multiclass"],
    "Logistic Regression": ["binary", "multiclass"],
    "SVM": ["binary", "multiclass"],
    "Decision Tree": ["binary", "multiclass"],

    # regression-capable
    "XGBoost Regressor": ["continuous"],
    "Random Forest Regressor": ["continuous"],
    "Linear Regression": ["continuous"],
    "LightGBM Regressor": ["continuous"]
}

def validate_model_for_task(model_type: str, task_type: str) -> bool:
    """Return True if model_type is supported for the given task_type."""
    supported_tasks = MODEL_VALIDATION.get(model_type)
    if not supported_tasks:
        return False
    return task_type in supported_tasks


@router.post("/training/detect-task-type", response_model=DetectTaskTypeResponse ,include_in_schema=False)
@deprecated(reason="Use BiasGuard 2.0 monitoring workflow instead")
async def detect_task_type(request: DetectTaskTypeRequests):
    """
    Simple wrapper to detect task type (binary/multiclass/continuous)
    """
    try:
        # Get file from S3
        df = await get_file_from_s3(request.file_id)
        
        # Get target column
        if request.target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column {request.target_column} not found")
        
        y = df[request.target_column].dropna()
        
        # Detect type using your existing function
        task_type = type_of_target(y)
        
        # Get models for this type
        recommended_models = BEST_MODELS.get(task_type, ['XGBoost'])
        
        # Basic stats
        if task_type in ['binary', 'multiclass']:
            stats = {
                "unique_values": int(y.nunique()),
                "value_counts": y.value_counts().to_dict()
            }
        else:
            stats = {
                "min": float(y.min()),
                "max": float(y.max()),
                "mean": float(y.mean())
            }
        
        return DetectTaskTypeResponse(
            task_type=task_type,
            recommended_models=recommended_models,
            default_model=recommended_models[0],
            target_stats=stats
        )
        
    except Exception as e:
        logging.error(f"Task detection failed: {e}")
        raise HTTPException(status_code=410, detail="Training endpoint deprecated. Use BiasGuard 2.0 monitoring workflow.")

@router.post("/training", response_model=TrainingResponse, include_in_schema=False)
@deprecated(reason="Use BiasGuard 2.0 monitoring workflow instead")
async def train_model(request: TrainingRequest, db: Session = Depends(get_db)):
    """Train model with real-time progress updates"""
    
    model_id = f"model_{uuid.uuid4().hex[:8]}"
    
    # ✅ INITIALIZE progress tracking
    training_progress[model_id] = {
        "status": "initializing",
        "progress": 0,
        "stage": "Starting training...",
        "model_id": model_id
    }
    
    try:
        # Update: Data loading
        training_progress[model_id] = {
            "status": "preprocessing",
            "progress": 10,
            "stage": "Loading dataset..."
        }
        
        df = await get_file_from_s3(request.file_id)
        
        # Update: Validation
        training_progress[model_id] = {
            "status": "preprocessing",
            "progress": 20,
            "stage": "Validating data quality..."
        }
        
        validator = DataValidator(
            min_samples=500,
            min_group_size=30,
            max_missing_pct=15.0,
            max_class_imbalance=0.03,
            max_correlation_threshold=0.95
        )
        
        validation_result = validator.validate_dataset(
            df, 
            request.target_column, 
            request.sensitive_columns
        )
        
        if not validation_result["valid"]:
            training_progress[model_id] = {
                "status": "error",
                "progress": 0,
                "stage": "Validation failed",
                "error": str(validation_result["blockers"])
            }
            raise HTTPException(status_code=400, detail={
                "message": "Dataset quality issues detected",
                "blockers": validation_result["blockers"]
            })
        
        # Update: Preprocessing
        training_progress[model_id] = {
            "status": "preprocessing",
            "progress": 30,
            "stage": "Preprocessing data..."
        }
        
        X_train, X_test, y_train, y_test, s_train, s_test = preprocess_data(
            df, 
            request.target_column, 
            request.sensitive_columns,
            test_size=request.test_size
        )
        
        # Update: Task detection
        training_progress[model_id] = {
            "status": "preprocessing",
            "progress": 40,
            "stage": "Detecting task type..."
        }
        
        task_type = type_of_target(y_train)
        
        if request.model_type:
            if not validate_model_for_task(request.model_type, task_type):
                raise ValueError(f"Model incompatible with task type")
            model_choice = request.model_type
        else:
            model_choice = BEST_MODELS[task_type][0]
        
        # Update: Starting MLflow
        training_progress[model_id] = {
            "status": "training",
            "progress": 50,
            "stage": f"Training {model_choice}..."
        }
        
        mlflow.set_experiment("model_training")
        
        with mlflow.start_run(run_name=f"train_{model_choice}_{model_id}") as run:
            
            # Log params
            mlflow.log_params({
                "model_id": model_id,
                "model_type": model_choice,
                "task_type": task_type,
                "target_column": request.target_column,
                "sensitive_columns": str(request.sensitive_columns),
                "test_size": request.test_size,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": X_train.shape[1]
            })
            
            # Update: Model training
            training_progress[model_id] = {
                "status": "training",
                "progress": 60,
                "stage": "Fitting model on training data..."
            }
            
            # Train model
            if task_type in ["binary", "multiclass"]:
                model, y_pred = await train_classification(
                    X_train, X_test, y_train, y_test, model_choice
                )
                metrics = calculate_classification_metrics(y_test, y_pred)
            else:
                model, y_pred = await train_regression(
                    X_train, X_test, y_train, y_test, model_choice
                )
                metrics = calculate_regression_metrics(y_test, y_pred)
            
            # Update: Evaluation
            training_progress[model_id] = {
                "status": "evaluating",
                "progress": 80,
                "stage": "Evaluating model performance..."
            }
            
            mlflow.log_metrics(metrics)
            
            # Update: Saving
            training_progress[model_id] = {
                "status": "saving",
                "progress": 90,
                "stage": "Saving model and artifacts..."
            }
            
            mlflow.sklearn.autolog(disable=True)  
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=X_train[:5]
            )
            
            # Save test data
            with tempfile.TemporaryDirectory() as tmpdir:
                test_data_dir = f"{tmpdir}/test_data"
                os.makedirs(test_data_dir, exist_ok=True)
                
                np.save(f"{test_data_dir}/X_test.npy", X_test)
                np.save(f"{test_data_dir}/y_test.npy", y_test)
                np.save(f"{test_data_dir}/y_pred.npy", y_pred)
                np.save(f"{test_data_dir}/s_test.npy", s_test)
                np.save(f"{test_data_dir}/X_train.npy", X_train)
                np.save(f"{test_data_dir}/y_train.npy", y_train)
                np.save(f"{test_data_dir}/s_train.npy", s_train)
                
                metadata = {
                    "model_id": model_id,
                    "task_type": task_type,
                    "model_type": model_choice,
                    "target_column": request.target_column,
                    "sensitive_columns": request.sensitive_columns,
                    "feature_count": X_train.shape[1]
                }
                
                with open(f"{test_data_dir}/metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                mlflow.log_artifacts(test_data_dir, "test_data")
            
            time.sleep(0.5)
            
            # Store in RAM
            accuracy = metrics.get("accuracy", metrics.get("r2_score", 0.0))
            model_storage[model_id] = {
                "model": model,
                "X_train": X_train,
                "y_train": y_train,
                "s_train": s_train,
                "X_test": X_test,
                "y_test": y_test,
                "y_pred": y_pred,
                "s_test": s_test,
                "task_type": task_type,
                "model_type": model_choice,
                "target_column": request.target_column,
                "sensitive_columns": request.sensitive_columns,
                "metrics": metrics,
                "file_id": request.file_id,
                "mlflow_run_id": run.info.run_id
            }
            
            # Handle sensitive data
            if isinstance(request.sensitive_columns, list) and len(request.sensitive_columns) > 1:
                s_test_dict = {}
                for i, col_name in enumerate(request.sensitive_columns):
                    s_test_dict[col_name] = s_test[:, i]
                model_storage[model_id]["sensitive_data"] = s_test_dict
            else:
                col_name = request.sensitive_columns[0] if isinstance(request.sensitive_columns, list) else request.sensitive_columns
                model_storage[model_id]["sensitive_data"] = {
                    col_name: s_test.flatten() if s_test.ndim > 1 else s_test
                }
        
        # Update: Saving to database
        training_progress[model_id] = {
            "status": "saving",
            "progress": 95,
            "stage": "Saving to database..."
        }
        
        model_data_for_db = schemas.ModelCreate(
            model_id=model_id,
            model_type=model_choice,
            task_type=task_type,
            dataset_name=request.file_id,
            target_column=request.target_column,
            sensitive_columns=request.sensitive_columns,
            feature_count=X_train.shape[1],
            training_samples=len(X_train),
            test_samples=len(X_test),
            accuracy=float(metrics["accuracy"]),
            mlflow_run_id=run.info.run_id
        )   
        
        db_model = crud.create_model(db, model_data_for_db)
        
        # ✅ COMPLETE!
        training_progress[model_id] = {
            "status": "complete",
            "progress": 100,
            "stage": "Training complete!",
            "model_id": model_id
        }
        
        logging.info(f"✅ Training complete for {model_id}")
        
        return TrainingResponse(
            model_id=model_id,
            task_type=task_type,
            model_type=model_choice,
            metrics=metrics,
            training_samples=len(X_train),
            testing_samples=len(X_test),
            message=f"{model_choice} trained successfully"
        )
        
    except Exception as e:
        # ✅ Update progress on error
        training_progress[model_id] = {
            "status": "error",
            "progress": 0,
            "stage": "Training failed",
            "error": str(e)
        }
        logging.error(f"Training failed: {e}")
        raise HTTPException(status_code=410, detail="Training endpoint deprecated. Use BiasGuard 2.0 monitoring workflow.")

@router.get("/training/stream/{model_id}",include_in_schema=False)
async def stream_training_progress(model_id: str):
    """SSE endpoint for real-time training progress"""
    
    async def event_generator():
        last_progress = -1
        max_wait = 120
        elapsed = 0
        
        while elapsed < max_wait:
            status = training_progress.get(model_id)
            
            if not status:
                yield f"data: {json.dumps({'status': 'waiting', 'progress': 0})}\n\n"
                await asyncio.sleep(0.5)
                elapsed += 0.5
                continue
            
            current_progress = status.get('progress', 0)
            
            if current_progress != last_progress:
                yield f"data: {json.dumps(status)}\n\n"
                last_progress = current_progress
            
            if status.get('status') in ['complete', 'error']:
                yield f"data: {json.dumps({**status, 'done': True})}\n\n"
                break
            
            await asyncio.sleep(0.2)
            elapsed += 0.2
        
        yield f"data: {json.dumps({'status': 'timeout', 'done': True})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
    
      
def calculate_fairness_weights(y_train, s_train, strategy="reweighing"):
    """
    Calculate sample weights for fairness
    """
    n_samples = len(y_train)
    weights = np.ones(n_samples)
    
    if strategy == "reweighing":
        # Calculate weights to balance outcomes across sensitive groups
        for s_val in np.unique(s_train):
            for y_val in np.unique(y_train):
                mask = (s_train == s_val) & (y_train == y_val)
                n_group = np.sum(mask)
                
                if n_group > 0:
                    # Expected proportion if independent
                    p_s = np.sum(s_train == s_val) / n_samples
                    p_y = np.sum(y_train == y_val) / n_samples
                    expected_prop = p_s * p_y
                    
                    # Actual proportion
                    actual_prop = n_group / n_samples
                    
                    # Weight to achieve fairness
                    if actual_prop > 0:
                        weight = expected_prop / actual_prop
                        weights[mask] = weight
    
    elif strategy == "demographic_parity":
        # Weights to achieve equal positive rates across groups
        for s_val in np.unique(s_train):
            mask = (s_train == s_val)
            group_pos_rate = np.mean(y_train[mask])
            overall_pos_rate = np.mean(y_train)
            
            if group_pos_rate > 0 and group_pos_rate < 1:
                # Adjust weights for this group
                pos_mask = mask & (y_train == 1)
                neg_mask = mask & (y_train == 0)
                
                if group_pos_rate < overall_pos_rate:
                    # Upweight positives
                    weights[pos_mask] *= (overall_pos_rate / group_pos_rate)
                else:
                    # Upweight negatives
                    weights[neg_mask] *= ((1 - overall_pos_rate) / (1 - group_pos_rate))
    
    # Normalize weights
    weights = weights * n_samples / np.sum(weights)
    
    return weights

async def train_classification(X_train, X_test, y_train, y_test, model_type: str):
    """Train classification models"""
    
    if model_type == "XGBoost":
        from xgboost import XGBClassifier
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        model = XGBClassifier(
            scale_pos_weight=weights[1]/weights[0] if len(classes) == 2 else 1,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    elif model_type == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    elif model_type == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    elif model_type == "SVM":
        from sklearn.svm import SVC
        model = SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
    elif model_type == "Decision Tree":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        )
    else:
        # Default to XGBoost
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, y_pred

async def train_regression(X_train, X_test, y_train, y_test, model_type: str):
    """Train regression models"""
    
    if model_type == "XGBoost Regressor":
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    elif model_type == "Random Forest Regressor":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    elif model_type == "Linear Regression":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif model_type == "LightGBM Regressor":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,  
                max_depth=-1,   
                n_jobs=-1,
                random_state=42,
                verbose=-1)
    else:
        # Default to XGBoost
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, y_pred

def calculate_classification_metrics(y_true, y_pred):
    """Calculate classification metrics"""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, average='weighted')),
        "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average='weighted'))
    }

def calculate_regression_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2_score": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))
    }

@router.get("/training/models/available", include_in_schema=False)
async def get_available_models(task_type: Optional[str] = None):
    """Get list of available models for task type"""
    if task_type:
        return {
            "task_type": task_type,
            "models": BEST_MODELS.get(task_type, [])
        }
    return {"all_models": BEST_MODELS}

# Helper for bias detection
def get_model_data(model_id: str, db: Session = None) -> Dict:
    """
    Get model data from RAM cache or load from MLflow artifacts
    
    Args:
        model_id: Model identifier
        db: Database session (optional, needed for MLflow loading)
        
    Returns:
        Model data dict with all training/test data
    """
    
    if model_id in model_storage:
        logging.info(f"Loading model {model_id} from RAM cache")
        return model_storage[model_id]
    
    logging.info(f"📥 Model {model_id} not in cache, loading from MLflow...")
    
    if db is None:
        raise ValueError(f"Database session required to load model {model_id} from MLflow")
    
    
    db_model = crud.get_model(db, model_id)
    if not db_model:
        raise ValueError(f"Model {model_id} not found in database")
    
    mlflow_run_id = db_model.mlflow_run_id
    if not mlflow_run_id or mlflow_run_id == "":
        raise ValueError(f"Model {model_id} has no MLflow run ID (old model trained before MLflow integration)")
    
    try:
        
        client = MlflowClient()
        
        logging.info(f"Downloading artifacts for run {mlflow_run_id}")
        artifact_path = client.download_artifacts(mlflow_run_id, "test_data")
        
        # Load numpy arrays
        X_test = np.load(f"{artifact_path}/X_test.npy", allow_pickle=True)
        y_test = np.load(f"{artifact_path}/y_test.npy", allow_pickle=True)
        y_pred = np.load(f"{artifact_path}/y_pred.npy", allow_pickle=True)
        s_test = np.load(f"{artifact_path}/s_test.npy", allow_pickle=True)
        X_train = np.load(f"{artifact_path}/X_train.npy", allow_pickle=True)
        y_train = np.load(f"{artifact_path}/y_train.npy", allow_pickle=True)
        s_train = np.load(f"{artifact_path}/s_train.npy", allow_pickle=True)
        
        logging.info("Numpy arrays loaded from MLflow")
        
      
        model_uri = f"runs:/{mlflow_run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        logging.info("Model object loaded from MLflow")
        
     
        with open(f"{artifact_path}/metadata.json", 'r') as f:
            metadata = json.load(f)
      
        model_data = {
            "model": model,
            "X_train": X_train,
            "y_train": y_train,
            "s_train": s_train,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "s_test": s_test,
            "task_type": db_model.task_type,
            "model_type": db_model.model_type,
            "target_column": db_model.target_column,
            "sensitive_columns": db_model.sensitive_columns,
            "metrics": {"accuracy": db_model.accuracy},
            "file_id": db_model.dataset_name,
            "mlflow_run_id": mlflow_run_id
        }
        
 
        if isinstance(db_model.sensitive_columns, list) and len(db_model.sensitive_columns) > 1:
            s_test_dict = {}
            for i, col_name in enumerate(db_model.sensitive_columns):
                s_test_dict[col_name] = s_test[:, i]
            model_data["sensitive_data"] = s_test_dict
        else:
            col_name = db_model.sensitive_columns[0] if isinstance(db_model.sensitive_columns, list) else db_model.sensitive_columns
            model_data["sensitive_data"] = {col_name: s_test.flatten() if s_test.ndim > 1 else s_test}
      
        model_storage[model_id] = model_data
        
        logging.info(f"Model {model_id} loaded from MLflow and cached in RAM")
        
        return model_data
        
    except Exception as e:
        logging.error(f"Failed to load model from MLflow: {e}")
        raise ValueError(f"Could not load model {model_id} from MLflow: {str(e)}")


def store_model_data(model_id: str, model_data: Dict):
    """Store model data in the global model_storage dictionary"""
    model_storage[model_id] = model_data

