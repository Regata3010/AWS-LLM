from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

# Model Schemas
class ModelBase(BaseModel):
    model_id: str
    model_type: str
    task_type: str
    dataset_name: str
    target_column: str
    sensitive_columns: List[str]
    feature_count: int
    training_samples: int
    test_samples: int
    accuracy: float

class ModelCreate(ModelBase):
    mlflow_run_id: str

class ModelResponse(ModelBase):
    bias_status: str
    created_at: datetime
    updated_at: datetime
    mlflow_run_id: str
    
    class Config:
        from_attributes = True

# Bias Analysis Schemas
class BiasAnalysisCreate(BaseModel):
    analysis_id: str
    model_id: str
    compliance_status: str
    bias_status: str
    fairness_metrics: Dict[str, Any]
    aif360_metrics: Optional[Dict[str, Any]]
    recommendations: List[str]
    mlflow_run_id: str

class BiasAnalysisResponse(BiasAnalysisCreate):
    analyzed_at: datetime
    
    class Config:
        from_attributes = True

# Mitigation Schemas
class MitigationCreate(BaseModel):
    mitigation_id: str
    original_model_id: str
    new_model_id: str
    strategy: str
    original_accuracy: float
    new_accuracy: float
    original_disparate_impact: Optional[float]
    new_disparate_impact: Optional[float]
    accuracy_impact: float
    bias_improvement: Dict[str, Any]
    mlflow_run_id: str

class MitigationResponse(MitigationCreate):
    created_at: datetime
    
    class Config:
        from_attributes = True