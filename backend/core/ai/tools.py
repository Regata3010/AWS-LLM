# backend/core/ai/tools.py
"""
BiasGuard Tools for LangGraph Agent (CAG)
Standalone functions that work with LangChain's @tool decorator
"""

from langchain.tools import tool
from sqlalchemy.orm import Session
from typing import Optional
import json
from datetime import datetime, timedelta
from api.models.database import ExternalModel, PredictionLog, BiasAnalysis, Model
from api.models.user import User

def create_tools_for_user(db: Session, user: User):
    """
    Create tools with db and user context bound via closure
    This avoids the 'self' parameter conflict
    """
    
    @tool
    def get_user_models() -> str:
        """Get all models accessible to current user"""
        if user.is_superuser:
            models = db.query(ExternalModel).all()
        else:
            models = db.query(ExternalModel).filter(
                ExternalModel.organization_id == user.organization_id
            ).all()
        
        result = {"total_models": len(models), "models": []}
        
        for model in models:
            latest_analysis = db.query(BiasAnalysis).filter(
                BiasAnalysis.model_id == model.model_id
            ).order_by(BiasAnalysis.analyzed_at.desc()).first()
            
            result["models"].append({
                "model_id": model.model_id,
                "model_name": model.model_name,
                "framework": model.framework,
                "bias_status": latest_analysis.bias_status if latest_analysis else "unknown",
                "predictions_logged": db.query(PredictionLog).filter(
                    PredictionLog.model_id == model.model_id
                ).count(),
                "created_at": model.created_at.isoformat()
            })
        
        return json.dumps(result, indent=2)
    
    @tool
    def get_model_status(model_id: str) -> str:
        """Get detailed status of a specific model"""
        model = db.query(ExternalModel).filter(
            ExternalModel.model_id == model_id
        ).first()
        
        if not model:
            return json.dumps({"error": "Model not found"})
        
        if not user.is_superuser and model.organization_id != user.organization_id:
            return json.dumps({"error": "Access denied"})
        
        latest_analysis = db.query(BiasAnalysis).filter(
            BiasAnalysis.model_id == model_id
        ).order_by(BiasAnalysis.analyzed_at.desc()).first()
        
        pred_count = db.query(PredictionLog).filter(
            PredictionLog.model_id == model_id
        ).count()
        
        result = {
            "model_id": model_id,
            "model_name": model.model_name,
            "model_type": model.model_type,
            "framework": model.framework,
            "sensitive_attributes": model.sensitive_attributes,
            "predictions_logged": pred_count,
            "monitoring_enabled": model.monitoring_enabled,
            "created_at": model.created_at.isoformat(),
            "latest_analysis": None
        }
        
        if latest_analysis:
            metrics = latest_analysis.fairness_metrics if isinstance(latest_analysis.fairness_metrics, dict) else json.loads(latest_analysis.fairness_metrics or '{}')
            
            result["latest_analysis"] = {
                "analysis_id": latest_analysis.analysis_id,
                "bias_status": latest_analysis.bias_status,
                "compliance_status": latest_analysis.compliance_status,
                "fairness_metrics": metrics,
                "analyzed_at": latest_analysis.analyzed_at.isoformat(),
                "recommendations": latest_analysis.recommendations
            }
        
        return json.dumps(result, indent=2)
    
    @tool
    def identify_violations(model_id: str) -> str:
        """Identify specific regulation violations for a model"""
        latest_analysis = db.query(BiasAnalysis).filter(
            BiasAnalysis.model_id == model_id
        ).order_by(BiasAnalysis.analyzed_at.desc()).first()
        
        if not latest_analysis:
            return json.dumps({"error": "No analysis found"})
        
        metrics = latest_analysis.fairness_metrics if isinstance(latest_analysis.fairness_metrics, dict) else json.loads(latest_analysis.fairness_metrics or '{}')
        
        violations = []
        
        for attr, m in metrics.items():
            if 'disparate_impact' in m:
                di = m['disparate_impact'].get('ratio', 1.0)
                if isinstance(di, (int, float)) and (di < 0.8 or di > 1.25):
                    violations.append({
                        "regulation": "ECOA Section 1002.6(a)",
                        "metric": "disparate_impact",
                        "attribute": attr,
                        "current_value": di,
                        "threshold": "0.8-1.25",
                        "severity": "critical" if di < 0.8 else "warning",
                        "description": f"Disparate impact for {attr} is {di:.3f}",
                        "potential_penalty": "$10,000 - $500,000",
                        "case_law": "Griggs v. Duke Power Co. (1971)"
                    })
            
            if 'statistical_parity' in m:
                sp = abs(m['statistical_parity'].get('statistical_parity_diff', 0))
                if sp > 0.1:
                    violations.append({
                        "regulation": "Title VII Civil Rights Act",
                        "metric": "statistical_parity",
                        "attribute": attr,
                        "current_value": sp,
                        "threshold": "0.1",
                        "severity": "critical",
                        "description": f"Statistical parity difference for {attr} is {sp:.3f}"
                    })
        
        result = {
            "model_id": model_id,
            "total_violations": len(violations),
            "violations": violations,
            "compliance_status": "COMPLIANT" if len(violations) == 0 else "NON-COMPLIANT"
        }
        
        return json.dumps(result, indent=2)
    
    # Return list of tool functions
    return [get_user_models, get_model_status, identify_violations]

# Renamed to match usage
def get_tools_for_user(db: Session, user: User) -> list:
    """Get all CAG tools for user"""
    return create_tools_for_user(db, user)