from pydantic import BaseModel, Field, field_validator,EmailStr
from typing import Optional
from typing import List,Optional,Dict

class DatasetUploadRequests(BaseModel):
   pass

class ColumnAnalysisRequests(BaseModel):
    file_id:str
    use_llm: bool = True  # Add this
    domain: str = "finance"

class ColumnOverrideRequests(BaseModel):
   file_id:str
   target_column:str
   sensitive_columns:List[str]

class TrainingRequest(BaseModel):
    file_id: str
    target_column: str
    sensitive_columns: List[str]
    model_type: Optional[str] = None  # Optional - auto-selects if not provided
    test_size: float = 0.2
    
    enable_fairness: bool = False
    fairness_strategy: Optional[str] = None
    
    # class Config:
    #     schema_extra = {
    #         "example": {
    #             "file_id": "uploads/dataset_123.csv",
    #             "target_column": "income",
    #             "sensitive_columns": ["age", "gender"],
    #             "model_type": "XGBoost",
    #             "test_size": 0.2,
    #             "enable_fairness": False,
    #             "fairness_strategy": None
    #         }
    #     }
    
class BiasDetectionRequest(BaseModel):
    model_id: str 
    
class DetectTaskTypeRequests(BaseModel):
    file_id: str
    target_column: str
    
class MitigationRequest(BaseModel):
    model_id: str = Field(..., description="ID of the model to mitigate")
    mitigation_strategy: str = Field(
        default="auto",
        description="Mitigation strategy",
        enum=["auto", "reweighing", "threshold_optimization", "fairness_constraints"]
    )
    max_accuracy_loss: float = Field(
        default=0.05,
        description="Maximum acceptable accuracy loss (default 5%)"
    )
    
    # class Config:
    #     schema_extra = {
    #         "example": {
    #             "model_id": "model_abc123",
    #             "mitigation_strategy": "auto",
    #             "max_accuracy_loss": 0.05
    #         }
    #     }
    
class ColumnSelectionRequest(BaseModel):
    """Request model for column selection - LLM generates the columns"""
    file_id: str = Field(..., description="Unique identifier for uploaded dataset")
    use_llm: bool = Field(default=True, description="Use LLM for auto-detection")
    domain: Optional[str] = Field(default="finance", description="Domain context for bias analysis")
    
    
class ReportGenerateRequest(BaseModel):
    report_type: str = "compliance"  # compliance, executive, technical
    include_recommendations: bool = True
    include_technical_details: bool = True
    format: str = "pdf"  # pdf or html
    

class PredictionLogRequest(BaseModel):
    """Log a single prediction"""
    model_id: str
    prediction: int
    prediction_proba: Optional[float] = None
    ground_truth: Optional[int] = None
    features: Optional[Dict] = None
    sensitive_attributes: Dict  # REQUIRED: {race: "White", gender: "Male", age: 35}
    
    
class BatchPredictionLogRequest(BaseModel):
    """Log multiple predictions via API"""
    model_id: str
    predictions: List[int]
    prediction_probas: Optional[List[float]] = None
    ground_truths: Optional[List[int]] = None
    features: Optional[List[Dict]] = None
    sensitive_attributes: List[Dict]  # REQUIRED
    
    @field_validator('predictions')
    def validate_predictions(cls, v):
        if len(v) == 0:
            raise ValueError("predictions cannot be empty")
        return v
    
    @field_validator('sensitive_attributes')
    def validate_sensitive_attrs(cls, v, values):
        if 'predictions' in values and len(v) != len(values['predictions']):
            raise ValueError("sensitive_attributes must match predictions length")
        return v
    
class ModelRegisterRequest(BaseModel):
    """Register an external model"""
    model_name: str
    description: Optional[str] = None
    model_type: str = "classification"
    framework: Optional[str] = None
    version: Optional[str] = "v1.0"
    endpoint_url: Optional[str] = None
    sensitive_attributes: List[str]
    alert_thresholds: Optional[dict] = None

class AnalyzeRequest(BaseModel):
    """Request bias analysis on logged predictions"""
    model_id: str
    period_days: Optional[int] = 30
    min_samples: Optional[int] = 100
    

class UserRegister(BaseModel):
    username: str           # ✅ Added username
    email: EmailStr
    password: str
    organization_name: str
    
class ChatRequest(BaseModel):
    message: str
    model_id: Optional[str] = None  # Optional context
    thread_id: Optional[str] = None 