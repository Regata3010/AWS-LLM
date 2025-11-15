from typing import List, Optional, Dict, Any 
from pydantic import BaseModel

class DatasetUploadResponse(BaseModel):
    file_id: str
    filename: str
    rows: int
    columns: int
    s3_path: str
    upload_timestamp: str
    file_size_mb: float
    # column_names : List[str]
    
class TokenUsage(BaseModel):
    """Token usage breakdown"""
    input: int  # Prompt tokens
    output: int  # Completion tokens
    total: int  # Total tokens
    estimated_input: Optional[int] = None  # For comparison
    cost_usd: Optional[float] = None
    model : Optional[str] = None
    
class ColumnAnalysisResponse(BaseModel):
    file_id: str
    suggested_target: Optional[str]
    suggested_sensitive: List[str]
    all_columns: List[str]
    column_metadata: Dict[str, Dict]
    llm_analysis: Optional[Dict]
    manual_mode: bool
    
class ColumnOverrideResponse(BaseModel):
    file_id: str
    target_column: str
    sensitive_columns: List[str]
    validation: Dict
    ready_for_analysis: bool
    
class TrainingResponse(BaseModel):
    model_id: str
    task_type: str
    model_type: str
    metrics: Dict
    training_samples: int
    testing_samples: int
    message: str # was till here 
    data_validation_warnings: Optional[List[str]] = None  # User sees warnings
    data_quality_metadata: Optional[Dict[str, Any]] = None
    
class BiasDetectionResponse(BaseModel):
    model_id: str
    mlflow_run_id: str
    fairness_metrics: Optional[Dict]
    aif360_metrics: Optional[Dict]
    compliance_status: str
    recommendations: List[str]
    task_type: str
    latency:float
    fairness_applied : Optional[bool] = False
    intersectionality_analysis: Optional[Dict] = None
    
class DetectTaskTypeResponse(BaseModel):
    task_type: str  
    recommended_models: List[str]
    default_model: str
    target_stats: Dict[str, Any]
    
class MitigationResponse(BaseModel):
    status: str
    original_model_id: str
    new_model_id: str
    mlflow_run_id: str
    strategy_applied: str
    original_metrics: Dict[str, float]
    new_metrics: Dict[str, float]
    bias_improvement: Dict[str, float]
    accuracy_impact: float
    message: str
    
    validation_score: Optional[float] = None  # 0-100 score
    validation_success: Optional[bool] = None  # True if mitigation worked
    validation_issues: Optional[List[str]] = None  # Critical problems
    validation_warnings: Optional[List[str]] = None  # Non-critical concerns
    validation_recommendations: Optional[List[str]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "original_model_id": "model_abc123",
                "new_model_id": "model_def456",
                "mlflow_run_id": "run_789xyz",
                "strategy_applied": "reweighing",
                "original_metrics": {
                    "accuracy": 0.85,
                    "disparate_impact": 0.65,
                    "statistical_parity_diff": 0.18
                },
                "new_metrics": {
                    "accuracy": 0.83,
                    "disparate_impact": 0.88,
                    "statistical_parity_diff": 0.09
                },
                "bias_improvement": {
                    "disparate_impact": 0.23,
                    "statistical_parity_diff": 0.09
                },
                "accuracy_impact": -0.02,
                "message": "Mitigation fully successful (Score: 95/100)",
                "validation_score": 95.0,
                "validation_success": True,
                "validation_issues": [],
                "validation_warnings": [],
                "validation_recommendations": [
                    "Mitigation successful - model ready for deployment"
                ]
            }
        }
class ColumnSelectionResponse(BaseModel):
    file_id: str
    all_columns: List[str]
    column_metadata: Dict[str, Dict]
    analysis_time_seconds: float
    tokens: Optional[Dict[str, Any]] = None
    # LLM/Heuristic suggestions
    suggested_target: Optional[str] = None
    suggested_sensitive: Optional[List[str]] = None
    target_confidence: Optional[float] = None
    target_rationale: Optional[str] = None
    sensitive_details: Optional[List[Dict]] = None
    llm_warnings: Optional[List[str]] = None
    
    # User manual selection (confirmed)
    selected_target: Optional[str] = None
    selected_sensitive: Optional[List[str]] = None
    
    # Advanced analysis fields
    proxy_variables: Optional[List[Dict]] = None
    proxy_detection_summary: Optional[Dict] = None # NEW: Summary with counts and recommendations
    
    intersectionality_analysis: Optional[Dict] = None
    compliance_notes: Optional[Dict] = None
    recommended_metrics: Optional[List[str]] = None
    full_llm_analysis: Optional[Dict] = None
    
    # Validation results
    validation: Dict[str, Any]
    detection_method: str  # 'llm', 'heuristic', or 'manual'
    ready_for_training: bool
    
    data_quality_issues: Optional[List[str]] = None
    
    
class MitigationInfoResponse(BaseModel):
    """Response for GET /model/{id}/mitigation - separate from POST mitigation"""
    original_model_id: str
    strategy: str
    original_accuracy: float
    new_accuracy: float
    accuracy_impact: float
    bias_improvement: Dict[str, float]
    created_at: str
    mlflow_run_id: str
    

class ReportGenerateResponse(BaseModel):
    report_id: str
    download_url: str
    generated_at: str
    model_id: str
    report_type: str
    file_size_mb: float
    llm_summary: Optional[str] = None
    
    

class MonitoringStatsResponse(BaseModel):
    """Monitoring statistics"""
    model_id: str
    period_days: int
    total_predictions: int
    approval_rate: float
    predictions_per_day: float
    latest_prediction: str
    has_ground_truth: bool
    
    
class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    organization_id: str
    organization_name: str | None = None
    is_superuser: bool
    is_active: bool

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse
    
class ChatResponse(BaseModel):
    response: str
    tools_used: list
    timestamp: str