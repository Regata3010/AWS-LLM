from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Dict, Optional, List
import hashlib
import json
import random
import numpy as np
from datetime import datetime, timedelta, timezone
from api.models.database import get_db, Model
from api.models import schemas
from core.src.logger import logging
import mlflow
import os
import joblib


router = APIRouter()

class ABTestManager:
    """Manage A/B testing for models"""
    
    def __init__(self, db: Session):
        self.db = db
        self.active_tests = {}  # Store active A/B tests
        
    def create_ab_test(
        self, 
        model_a_id: str, 
        model_b_id: str, 
        test_name: str,
        traffic_split: float = 0.5,
        success_metric: str = "accuracy",  #NEED TO BE MORE OPTIONS 
        min_sample_size: int = 1000
    ) -> Dict:
        """Create a new A/B test between two models"""
        
        # Validate models exist
        model_a = self.db.query(Model).filter(Model.model_id == model_a_id).first()
        model_b = self.db.query(Model).filter(Model.model_id == model_b_id).first()
        
        if not model_a or not model_b:
            raise ValueError("One or both models not found")
        
        test_id = f"ab_test_{hashlib.md5(f'{model_a_id}_{model_b_id}_{datetime.now(timezone.utc)}'.encode()).hexdigest()[:8]}"
        
        test_config = {
            "test_id": test_id,
            "test_name": test_name,
            "model_a": {
                "model_id": model_a_id,
                "model_type": model_a.model_type,
                "baseline_accuracy": model_a.accuracy
            },
            "model_b": {
                "model_id": model_b_id,
                "model_type": model_b.model_type,
                "baseline_accuracy": model_b.accuracy
            },
            "traffic_split": traffic_split,  # Percentage to model B
            "success_metric": success_metric,
            "min_sample_size": min_sample_size,
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "results": {
                "model_a": {
                    "predictions": 0,
                    "correct": 0,
                    "accuracy": 0,
                    "bias_violations": 0
                },
                "model_b": {
                    "predictions": 0,
                    "correct": 0,
                    "accuracy": 0,
                    "bias_violations": 0
                },
                "statistical_significance": None,
                "winner": None
            }
        }
        
        # Store in active tests
        self.active_tests[test_id] = test_config
        
        # Log to MLflow
        mlflow.set_experiment("ab_tests")
        with mlflow.start_run(run_name=test_name):
            mlflow.log_params({
                "test_id": test_id,
                "model_a_id": model_a_id,
                "model_b_id": model_b_id,
                "traffic_split": traffic_split,
                "success_metric": success_metric
            })
        
        return test_config
    
    def route_prediction(self, test_id: str, user_features: Dict) -> Dict:
        """Route prediction to appropriate model based on A/B test configuration"""
    
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test_config = self.active_tests[test_id]
        
        if test_config["status"] != "active":
            raise ValueError(f"Test {test_id} is not active")
        
        # Hash user features for consistent routing
        feature_str = json.dumps(user_features, sort_keys=True)
        hash_value = int(hashlib.md5(feature_str.encode()).hexdigest(), 16)
        
        # Determine which model to use
        use_model_b = (hash_value % 100) < (test_config["traffic_split"] * 100)
        
        if use_model_b:
            model_id = test_config["model_b"]["model_id"]
            model_group = "model_b"
        else:
            model_id = test_config["model_a"]["model_id"]
            model_group = "model_a"
        
        # Get model from database
        model_record = self.db.query(Model).filter(Model.model_id == model_id).first()
        if not model_record:
            raise ValueError(f"Model {model_id} not found in database")
        
        # For now, simulate prediction since we don't have actual model files loaded
        # In production, you would load the model file here:
        # model_path = f"models/{model_id}.pkl"
        # if os.path.exists(model_path):
        #     model = joblib.load(model_path)
        #     prediction = model.predict(user_features)
        
        # Simulated prediction for testing
        prediction = {
            "model_used": model_id,
            "model_group": model_group,
            "model_type": model_record.model_type,
            "prediction": random.choice([0, 1]),  # Simulated
            "confidence": random.uniform(0.6, 0.99),
            "test_id": test_id
        }
        
        # Update test results
        test_config["results"][model_group]["predictions"] += 1
        if random.random() > 0.3:  # Simulate 70% accuracy
            test_config["results"][model_group]["correct"] += 1
        
        # Update accuracy
        if test_config["results"][model_group]["predictions"] > 0:
            test_config["results"][model_group]["accuracy"] = (
                test_config["results"][model_group]["correct"] / 
                test_config["results"][model_group]["predictions"]
            )
        
            return prediction
    
    def calculate_statistical_significance(self, test_id: str) -> Dict:
        """Calculate statistical significance of A/B test results"""
        
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test_config = self.active_tests[test_id]
        results = test_config["results"]
        
        # Extract metrics
        n_a = results["model_a"]["predictions"]
        n_b = results["model_b"]["predictions"]
        
        if n_a < test_config["min_sample_size"] or n_b < test_config["min_sample_size"]:
            return {
                "significant": False,
                "p_value": None,
                "confidence": 0,
                "message": f"Insufficient sample size. Need {test_config['min_sample_size']} samples per model"
            }
        
        # Calculate conversion rates
        p_a = results["model_a"]["accuracy"]
        p_b = results["model_b"]["accuracy"]
        
        # Pooled probability
        p_pool = (results["model_a"]["correct"] + results["model_b"]["correct"]) / (n_a + n_b)
        
        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        
        # Z-score
        if se > 0:
            z_score = (p_b - p_a) / se
            # Two-tailed p-value (simplified)
            p_value = 2 * (1 - min(0.9999, 0.5 * (1 + np.sign(z_score) * min(abs(z_score) / 4, 0.9999))))
        else:
            z_score = 0
            p_value = 1
        
        # Determine significance
        significant = p_value < 0.05
        confidence = (1 - p_value) * 100
        
        # Determine winner
        winner = None
        if significant:
            winner = "model_b" if p_b > p_a else "model_a"
            test_config["results"]["winner"] = winner
        
        test_config["results"]["statistical_significance"] = {
            "p_value": round(p_value, 4),
            "z_score": round(z_score, 4),
            "confidence": round(confidence, 2),
            "significant": significant
        }
        
        return {
            "significant": significant,
            "p_value": round(p_value, 4),
            "confidence": round(confidence, 2),
            "winner": winner,
            "improvement": round((p_b - p_a) * 100, 2) if winner else None
        }
    
    def stop_test(self, test_id: str) -> Dict:
        """Stop an A/B test and finalize results"""
        
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test_config = self.active_tests[test_id]
        test_config["status"] = "completed"
        test_config["completed_at"] = datetime.now(timezone.utc).isoformat()
        
        # Calculate final significance
        significance = self.calculate_statistical_significance(test_id)
        
        # Log final results to MLflow
        mlflow.set_experiment("ab_tests")
        with mlflow.start_run(run_name=f"{test_config['test_name']}_final"):
            mlflow.log_metrics({
                "model_a_accuracy": test_config["results"]["model_a"]["accuracy"],
                "model_b_accuracy": test_config["results"]["model_b"]["accuracy"],
                "p_value": significance["p_value"] if significance["p_value"] else 1.0,
                "total_predictions": (
                    test_config["results"]["model_a"]["predictions"] + 
                    test_config["results"]["model_b"]["predictions"]
                )
            })
            if significance["winner"]:
                mlflow.log_param("winner", significance["winner"])
        
        return test_config

# Global manager instance
ab_manager = None

def get_ab_manager(db: Session = Depends(get_db)):
    global ab_manager
    if ab_manager is None:
        ab_manager = ABTestManager(db)
    return ab_manager

@router.post("/ab-test/create")
async def create_ab_test(
    model_a_id: str,
    model_b_id: str,
    test_name: str,
    traffic_split: float = 0.5,
    success_metric: str = "accuracy",
    min_sample_size: int = 1000,
    manager: ABTestManager = Depends(get_ab_manager)
):
    """Create a new A/B test between two models"""
    try:
        test_config = manager.create_ab_test(
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            test_name=test_name,
            traffic_split=traffic_split,
            success_metric=success_metric,
            min_sample_size=min_sample_size
        )
        
        return {
            "status": "success",
            "test": test_config
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/ab-test/{test_id}/predict")
async def route_prediction(
    test_id: str,
    features: Dict,
    manager: ABTestManager = Depends(get_ab_manager)
):
    """Route a prediction through an A/B test"""
    try:
        prediction = manager.route_prediction(test_id, features)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/ab-test/{test_id}/results")
async def get_test_results(
    test_id: str,
    manager: ABTestManager = Depends(get_ab_manager)
):
    """Get current results of an A/B test"""
    if test_id not in manager.active_tests:
        raise HTTPException(status_code=404, detail=f"Test {test_id} not found")
    
    test_config = manager.active_tests[test_id]
    significance = manager.calculate_statistical_significance(test_id)
    
    return {
        "test": test_config,
        "significance": significance,
        "recommendation": _get_recommendation(test_config, significance)
    }

def _get_recommendation(test_config: Dict, significance: Dict) -> str:
    """Generate recommendation based on test results"""
    if not significance["significant"]:
        return "Continue collecting data. Results not yet statistically significant."
    
    winner = significance["winner"]
    improvement = significance["improvement"]
    
    if winner == "model_b":
        return f"Model B shows {improvement}% improvement. Consider promoting to production."
    else:
        return f"Model A performs better. Model B shows {abs(improvement)}% degradation."

@router.post("/ab-test/{test_id}/stop")
async def stop_ab_test(
    test_id: str,
    manager: ABTestManager = Depends(get_ab_manager)
):
    """Stop an A/B test"""
    try:
        final_results = manager.stop_test(test_id)
        return {
            "status": "success",
            "message": f"Test {test_id} stopped",
            "final_results": final_results
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/ab-test/active")
async def list_active_tests(manager: ABTestManager = Depends(get_ab_manager)):
    """List all active A/B tests"""
    active_tests = [
        test for test in manager.active_tests.values() 
        if test["status"] == "active"
    ]
    
    return {
        "total": len(active_tests),
        "tests": active_tests
    }