from sqlalchemy.orm import Session
from . import database, schemas
from typing import List, Optional
import uuid

# Model Ops

def create_model(db: Session, model: schemas.ModelCreate) -> database.Model:
    """Create a new model in the database"""
    db_model = database.Model(**model.model_dump())
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model

def get_model(db: Session, model_id: str) -> Optional[database.Model]:
    """Get a model by ID"""
    return db.query(database.Model).filter(database.Model.model_id == model_id).first()

def get_all_models(db: Session, skip: int = 0, limit: int = 100, bias_status: Optional[str] = None) -> List[database.Model]:
    """Get all models with pagination"""
    return db.query(database.Model).order_by(database.Model.created_at.desc()).offset(skip).limit(limit).all()

def get_models_count(db: Session) -> int:
    """Get total count of models"""
    return db.query(database.Model).count()

def update_model_bias_status(db: Session, model_id: str, bias_status: str):
    """Update model's bias status"""
    db_model = get_model(db, model_id)
    if db_model:
        db_model.bias_status = bias_status
        db.commit()
        db.refresh(db_model)
    return db_model

def delete_model(db: Session, model_id: str) -> bool:
    """Delete a model"""
    db_model = get_model(db, model_id)
    if db_model:
        db.delete(db_model)
        db.commit()
        return True
    return False

# Bias Analysis Ops

def create_bias_analysis(db: Session, analysis: schemas.BiasAnalysisCreate) -> database.BiasAnalysis:
    """Create a new bias analysis"""
    db_analysis = database.BiasAnalysis(**analysis.model_dump())
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)
    return db_analysis

def get_latest_bias_analysis(db: Session, model_id: str) -> Optional[database.BiasAnalysis]:
    """Get the most recent bias analysis for a model"""
    return db.query(database.BiasAnalysis).filter(
        database.BiasAnalysis.model_id == model_id
    ).order_by(database.BiasAnalysis.analyzed_at.desc()).first()

def get_bias_history(db: Session, model_id: str) -> List[database.BiasAnalysis]:
    """Get all bias analyses for a model"""
    return db.query(database.BiasAnalysis).filter(
        database.BiasAnalysis.model_id == model_id
    ).order_by(database.BiasAnalysis.analyzed_at.desc()).all()

# Mitigation Ops

def create_mitigation(db: Session, mitigation: schemas.MitigationCreate) -> database.Mitigation:
    """Create a new mitigation record"""
    db_mitigation = database.Mitigation(**mitigation.model_dump())
    db.add(db_mitigation)
    db.commit()
    db.refresh(db_mitigation)
    return db_mitigation

def get_model_mitigations(db: Session, model_id: str) -> List[database.Mitigation]:
    """Get all mitigations for a model"""
    return db.query(database.Mitigation).filter(
        database.Mitigation.original_model_id == model_id
    ).order_by(database.Mitigation.created_at.desc()).all()

def get_recent_models(db: Session, limit: int = 10) -> List[database.Model]:
    """
    Get most recently created/updated models
    
    For dashboard "Recent Activity" section
    
    Args:
        db: Database session
        limit: Number of recent models to return
        
    Returns:
        List of models ordered by most recent first
    """
    return db.query(database.Model).order_by(
        database.Model.updated_at.desc()
    ).limit(limit).all()

def get_dashboard_stats(db: Session, organization_id: str = None) -> dict:
    """Get aggregated statistics for dashboard"""
    
    # Base query
    model_query = db.query(database.Model)
    
    # Filter by organization if provided (regular users)
    # If organization_id is None, show all (superusers)
    if organization_id:
        model_query = model_query.filter(database.Model.organization_id == organization_id)
    
    total_models = model_query.count()
    compliant = model_query.filter(database.Model.bias_status == "compliant").count()
    warning = model_query.filter(database.Model.bias_status == "warning").count()
    critical = model_query.filter(database.Model.bias_status == "critical").count()
    unknown = model_query.filter(database.Model.bias_status == "unknown").count()
    
    # Same for analyses
    analysis_query = db.query(database.BiasAnalysis)
    if organization_id:
        analysis_query = analysis_query.filter(database.BiasAnalysis.organization_id == organization_id)
    
    total_analyses = analysis_query.count()
    
    return {
        "total_models": total_models,
        "compliant_models": compliant,
        "models_at_risk": warning,
        "critical_models": critical,
        "unknown_status": unknown,
        "total_analyses": total_analyses,
        "compliance_rate": round((compliant / total_models * 100) if total_models > 0 else 0, 1)
    }