from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import uuid
from api.models.database import get_db, Model, Mitigation, ExternalModel, PredictionLog
from api.models import schemas, crud
from api.models.requests import ModelRegisterRequest
from core.src.logger import logging
from api.models.user import User
from core.auth.dependencies import get_current_active_user, require_admin
from core.cache.redis_client import get_redis

router = APIRouter()


# BIASGUARD 2.0 - MODEL REGISTRY


@router.post("/models/register")
async def register_external_model(
    request: ModelRegisterRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Register an external model for monitoring (BiasGuard 2.0)
    
    Use this to add models trained in SageMaker, Azure ML, Databricks, etc.
    """
    
    try:
        model_id = f"model_{uuid.uuid4().hex[:12]}"
        
        model = ExternalModel(
            model_id=model_id,
            model_name=request.model_name,
            description=request.description,
            model_type=request.model_type,
            framework=request.framework,
            version=request.version,
            endpoint_url=request.endpoint_url,
            sensitive_attributes=request.sensitive_attributes,
            monitoring_enabled=True,
            organization_id=current_user.organization_id,
            created_by= current_user.id,
            alert_thresholds=request.alert_thresholds or {
                "disparate_impact_min": 0.8,
                "disparate_impact_max": 1.25,
                "statistical_parity_max": 0.1
            },
            status="active"
        )
        
        db.add(model)
        db.commit()
        db.refresh(model)
        
        redis = get_redis()
        redis.delete(f"models:list:{current_user.organization_id}")
        redis.delete(f"dashboard:summary:{current_user.organization_id}")
        logging.info(f"Registered external model: {model_id} - {request.model_name}")
        
        return {
            "status": "success",
            "model_id": model_id,
            "model_name": request.model_name,
            "message": f"Model '{request.model_name}' registered successfully",
            "next_steps": [
                f"Upload predictions: POST /api/v1/monitor/upload_csv?model_id={model_id}",
                f"Analyze: POST /api/v1/analyze",
                f"View stats: GET /api/v1/monitor/stats/{model_id}"
            ]
        }
        
    except Exception as e:
        logging.error(f"Failed to register external model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# UNIFIED MODEL LISTING (Both Tables)


# @router.get("/models")
# async def list_all_models(
#     skip: int = 0,
#     limit: int = 100,
#     model_source: Optional[str] = None,  # "trained", "external", or None (all)
#     bias_status: Optional[str] = None,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_active_user)
# ):
#     """
#     List ALL models (BiasGuard 1.0 trained + BiasGuard 2.0 external)
    
#     Args:
#         model_source: Filter by source ("trained", "external", or None for all)
#         bias_status: Filter by bias status (for trained models only)
#     """
    
#     all_models = []
    
#     # Get external models (BiasGuard 2.0)
#     if model_source in [None, "external"]:
#         external_models = db.query(ExternalModel).filter(
#             ExternalModel.organization_id == current_user.organization_id 
#         ).order_by(ExternalModel.created_at.desc()).all()
    
#     if model_source in [None, "trained"]:
#         trained_models = db.query(Model).filter(
#             Model.organization_id == current_user.organization_id  # ADD THIS
#         ).order_by(Model.created_at.desc()).all()
        
#         for m in external_models:
#             # Count predictions
#             pred_count = db.query(PredictionLog).filter(
#                 PredictionLog.model_id == m.model_id
#             ).count()
            
#             # Get latest analysis
#             latest_analysis = crud.get_latest_bias_analysis(db, m.model_id)
            
#             all_models.append({
#                 "model_id": m.model_id,
#                 "model_name": m.model_name,
#                 "model_type": m.model_type or "classification",
#                 "framework": m.framework,
#                 "version": m.version,
#                 "source": "external",
#                 "status": m.status,
#                 "monitoring_enabled": m.monitoring_enabled,
#                 "sensitive_columns": m.sensitive_attributes,
#                 "predictions_logged": pred_count,
#                 "bias_status": latest_analysis.bias_status if latest_analysis else "unknown",
#                 "created_at": m.created_at.isoformat(),
#                 "updated_at": m.updated_at.isoformat(),
#                 "platform": "BiasGuard 2.0",
#                 "has_analysis": latest_analysis is not None
#             })
    
#     # Get trained models (BiasGuard 1.0 - legacy)
#     if model_source in [None, "trained"]:
#         trained_query = db.query(Model)
#         if bias_status:
#             trained_query = trained_query.filter(Model.bias_status == bias_status)
        
#         trained_models = trained_query.order_by(Model.created_at.desc()).all()
        
#         for model in trained_models:
#             # Check if mitigated
#             mitigation = db.query(Mitigation).filter(
#                 Mitigation.new_model_id == model.model_id
#             ).first()
            
#             all_models.append({
#                 "model_id": model.model_id,
#                 "model_name": model.model_type,
#                 "model_type": model.model_type,
#                 "task_type": model.task_type,
#                 "source": "trained",
#                 "accuracy": model.accuracy,
#                 "bias_status": model.bias_status or "unknown",
#                 "sensitive_columns": model.sensitive_columns,
#                 "training_samples": model.training_samples,
#                 "test_samples": model.test_samples,
#                 "dataset_name": model.dataset_name,
#                 "mlflow_run_id": model.mlflow_run_id,
#                 "is_mitigated": mitigation is not None,
#                 "created_at": model.created_at.isoformat(),
#                 "updated_at": model.updated_at.isoformat(),
#                 "platform": "BiasGuard 1.0 (Legacy)",
#                 "has_analysis": True  # Trained models always have analysis
#             })
    
#     # Sort by created_at (most recent first)
#     all_models.sort(key=lambda x: x['created_at'], reverse=True)
    
#     # Paginate
#     paginated = all_models[skip:skip+limit]
    
#     return {
#         "total": len(all_models),
#         "showing": len(paginated),
#         "skip": skip,
#         "limit": limit,
#         "breakdown": {
#             "external": len([m for m in all_models if m['source'] == 'external']),
#             "trained": len([m for m in all_models if m['source'] == 'trained'])
#         },
#         "models": paginated
#     }

@router.get("/models")
async def list_all_models(
    skip: int = 0,
    limit: int = 100,
    model_source: Optional[str] = None,
    bias_status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    List ALL models (BiasGuard 1.0 trained + BiasGuard 2.0 external)
    
    CACHED for 30 seconds (reduces DB load by 80%)
    
    Args:
        model_source: Filter by source ("trained", "external", or None for all)
        bias_status: Filter by bias status (for trained models only)
    """
    
    try:
        
        redis = get_redis()
        cache_key = f"models:list:{current_user.organization_id}:{skip}:{limit}:{model_source}:{bias_status}"
        
        # Try cache first
        cached = redis.get(cache_key)
        if cached:
            logging.info(f"Cache HIT: models list")
            return cached
        
        # Cache MISS - query database
        logging.info(f"Cache MISS: models list - Querying database")
        
        all_models = []
        
        # Get external models (BiasGuard 2.0)
        if model_source in [None, "external"]:
            external_models = db.query(ExternalModel).filter(
                ExternalModel.organization_id == current_user.organization_id 
            ).order_by(ExternalModel.created_at.desc()).all()
            
            for m in external_models:
                pred_count = db.query(PredictionLog).filter(
                    PredictionLog.model_id == m.model_id
                ).count()
                
                latest_analysis = crud.get_latest_bias_analysis(db, m.model_id)
                
                all_models.append({
                    "model_id": m.model_id,
                    "model_name": m.model_name,
                    "model_type": m.model_type or "classification",
                    "framework": m.framework,
                    "version": m.version,
                    "source": "external",
                    "status": m.status,
                    "monitoring_enabled": m.monitoring_enabled,
                    "sensitive_columns": m.sensitive_attributes,
                    "predictions_logged": pred_count,
                    "bias_status": latest_analysis.bias_status if latest_analysis else "unknown",
                    "created_at": m.created_at.isoformat(),
                    "updated_at": m.updated_at.isoformat(),
                    "platform": "BiasGuard 2.0",
                    "has_analysis": latest_analysis is not None
                })
        
        # Get trained models (BiasGuard 1.0)
        if model_source in [None, "trained"]:
            trained_query = db.query(Model).filter(
                Model.organization_id == current_user.organization_id
            )
            
            if bias_status:
                trained_query = trained_query.filter(Model.bias_status == bias_status)
            
            trained_models = trained_query.order_by(Model.created_at.desc()).all()
            
            for model in trained_models:
                mitigation = db.query(Mitigation).filter(
                    Mitigation.new_model_id == model.model_id
                ).first()
                
                all_models.append({
                    "model_id": model.model_id,
                    "model_name": model.model_type,
                    "model_type": model.model_type,
                    "task_type": model.task_type,
                    "source": "trained",
                    "accuracy": model.accuracy,
                    "bias_status": model.bias_status or "unknown",
                    "sensitive_columns": model.sensitive_columns,
                    "training_samples": model.training_samples,
                    "test_samples": model.test_samples,
                    "dataset_name": model.dataset_name,
                    "mlflow_run_id": model.mlflow_run_id,
                    "is_mitigated": mitigation is not None,
                    "created_at": model.created_at.isoformat(),
                    "updated_at": model.updated_at.isoformat(),
                    "platform": "BiasGuard 1.0 (Legacy)",
                    "has_analysis": True
                })
        
        all_models.sort(key=lambda x: x['created_at'], reverse=True)
        paginated = all_models[skip:skip+limit]
        
        response = {
            "total": len(all_models),
            "showing": len(paginated),
            "skip": skip,
            "limit": limit,
            "breakdown": {
                "external": len([m for m in all_models if m['source'] == 'external']),
                "trained": len([m for m in all_models if m['source'] == 'trained'])
            },
            "models": paginated
        }
        
       
        redis.set(cache_key, response, ttl=30)
        
        return response
        
    except Exception as e:
        logging.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/external")
async def list_external_models(
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List only external models (BiasGuard 2.0)"""
    
    query = db.query(ExternalModel).filter(
        ExternalModel.organization_id == current_user.organization_id  
    )
    
    if status:
        query = query.filter(ExternalModel.status == status)
    
    total = query.count()
    models = query.order_by(ExternalModel.created_at.desc()).offset(skip).limit(limit).all()
    
    return {
        "total": total,
        "showing": len(models),
        "models": [
            {
                "model_id": m.model_id,
                "model_name": m.model_name,
                "version": m.version,
                "framework": m.framework,
                "status": m.status,
                "monitoring_enabled": m.monitoring_enabled,
                "created_at": m.created_at.isoformat(),
                "sensitive_attributes": m.sensitive_attributes
            }
            for m in models
        ]
    }


# MODEL DETAILS (Works with Both)


@router.get("/model/{model_id}")
async def get_model_details(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get detailed information about ANY model (trained or external)
    """
    try:
        # Try ExternalModel first
        external_model = db.query(ExternalModel).filter(
            ExternalModel.model_id == model_id,
            ExternalModel.organization_id == current_user.organization_id
        ).first()
        
        if external_model:
            # Count predictions
            pred_count = db.query(PredictionLog).filter(
                PredictionLog.model_id == model_id
            ).count()
            
            # Get latest analysis
            latest_analysis = crud.get_latest_bias_analysis(db, model_id)
            
            return {
                "model_id": external_model.model_id,
                "model_name": external_model.model_name,
                "description": external_model.description,
                "model_type": external_model.model_type,
                "framework": external_model.framework,
                "version": external_model.version,
                "source": "external",
                "platform": "BiasGuard 2.0",
                "sensitive_attributes": external_model.sensitive_attributes,
                "monitoring_enabled": external_model.monitoring_enabled,
                "alert_thresholds": external_model.alert_thresholds,
                "status": external_model.status,
                "created_at": external_model.created_at.isoformat(),
                "updated_at": external_model.updated_at.isoformat(),
                "stats": {
                    "total_predictions_logged": pred_count,
                    "bias_status": latest_analysis.bias_status if latest_analysis else "unknown",
                    "latest_analysis": latest_analysis.analyzed_at.isoformat() if latest_analysis else None
                }
            }
        
        # Fallback to trained model
        model = crud.get_model(db, model_id)
        
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id} not found"
            )
        
        # Return trained model details (use existing schema)
        return schemas.ModelResponse.model_validate(model)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to get model details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/model/{model_id}")
async def delete_model(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin) 
):
    """
    Delete a model (works with both trained and external)
    """
    try:
        # Try ExternalModel first
        external_model = db.query(ExternalModel).filter(
            ExternalModel.model_id == model_id,
            ExternalModel.organization_id == current_user.organization_id
        ).first()
        
        if external_model:
            # Delete prediction logs
            db.query(PredictionLog).filter(
                PredictionLog.model_id == model_id
            ).delete()
            
            # Delete bias analyses
            from api.models.database import BiasAnalysis
            db.query(BiasAnalysis).filter(
                BiasAnalysis.model_id == model_id
            ).delete()
            
            # Delete model
            db.delete(external_model)
            db.commit()
            
            logging.info(f"Deleted external model {model_id}")
            
            return {
                "status": "success",
                "message": f"External model {model_id} deleted successfully"
            }
        
        # Fallback to trained model deletion
        deleted = crud.delete_model(db, model_id)
        
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id} not found"
            )
        
        logging.info(f"Deleted trained model {model_id}")
        
        return {
            "status": "success",
            "message": f"Trained model {model_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to delete model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# BIAS ANALYSIS ENDPOINTS (Works with Both)

@router.get("/bias/latest/{model_id}", response_model=schemas.BiasAnalysisResponse)
async def get_latest_bias_analysis(
    model_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the most recent bias analysis for ANY model
    Works with both BiasGuard 1.0 and 2.0 models
    """
    try:
        analysis = crud.get_latest_bias_analysis(db, model_id)
        
        if not analysis:
            raise HTTPException(
                status_code=404,
                detail=f"No bias analysis found for model {model_id}"
            )
        
        return schemas.BiasAnalysisResponse.model_validate(analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to get bias analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/bias/history/{model_id}")
async def get_bias_analysis_history(
    model_id: str,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    Get all bias analyses for a model (works with both platforms)
    """
    try:
        history = crud.get_bias_history(db, model_id)
        
        # Sort by analyzed_at (most recent first)
        history = sorted(history, key=lambda x: x.analyzed_at, reverse=True)[:limit]
        
        return {
            "model_id": model_id,
            "total_analyses": len(history),
            "history": [schemas.BiasAnalysisResponse.model_validate(a) for a in history]
        }
        
    except Exception as e:
        logging.error(f"Failed to get bias history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# MITIGATION INFO (Legacy Support)


@router.get("/model/{model_id}/mitigation")
async def get_model_mitigation_info(
    model_id: str,
    db: Session = Depends(get_db)
):
    """
    Get mitigation information for BiasGuard 1.0 trained models
    
    Note: BiasGuard 2.0 models don't support auto-mitigation.
    Use recommendations from analysis instead.
    """
    
    mitigation = db.query(Mitigation).filter(
        Mitigation.new_model_id == model_id
    ).first()
    
    if not mitigation:
        raise HTTPException(
            status_code=404,
            detail="No mitigation info found. This may be a BiasGuard 2.0 external model."
        )
    
    return {
        "original_model_id": mitigation.original_model_id,
        "strategy": mitigation.strategy,
        "original_accuracy": mitigation.original_accuracy,
        "new_accuracy": mitigation.new_accuracy,
        "accuracy_impact": mitigation.accuracy_impact,
        "bias_improvement": mitigation.bias_improvement,
        "created_at": mitigation.created_at.isoformat(),
        "mlflow_run_id": mitigation.mlflow_run_id,
        "platform": "BiasGuard 1.0 (Legacy)"
    }

# ========================================
# MODEL MANAGEMENT
# ========================================

@router.put("/models/{model_id}")
async def update_model(
    model_id: str,
    model_name: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    monitoring_enabled: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Update external model metadata (BiasGuard 2.0 only)"""
    
    model = db.query(ExternalModel).filter(
        ExternalModel.model_id == model_id
    ).first()
    
    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"External model {model_id} not found. Legacy models cannot be updated."
        )
    
    # Update fields
    if model_name:
        model.model_name = model_name
    if description is not None:
        model.description = description
    if status:
        model.status = status
    if monitoring_enabled is not None:
        model.monitoring_enabled = monitoring_enabled
    
    model.updated_at = datetime.utcnow()
    
    db.commit()
    
    return {
        "status": "success",
        "model_id": model_id,
        "message": "Model updated successfully"
    }

# ========================================
# STATISTICS & INSIGHTS
# ========================================

@router.get("/models/stats")
async def get_platform_stats(db: Session = Depends(get_db)):
    """
    Get platform-wide statistics
    Shows BiasGuard 1.0 vs 2.0 usage
    """
    
    # Count models
    external_count = db.query(ExternalModel).count()
    trained_count = db.query(Model).count()
    
    # Count active monitoring
    active_external = db.query(ExternalModel).filter(
        ExternalModel.monitoring_enabled == True,
        ExternalModel.status == "active"
    ).count()
    
    # Count predictions logged
    total_predictions = db.query(PredictionLog).count()
    
    # Count analyses
    from api.models.database import BiasAnalysis
    total_analyses = db.query(BiasAnalysis).count()
    
    # Models at risk
    critical_models = db.query(Model).filter(
        Model.bias_status == "critical"
    ).count()
    
    warning_models = db.query(Model).filter(
        Model.bias_status == "warning"
    ).count()
    
    return {
        "platform": "BiasGuard",
        "version": "2.0",
        "models": {
            "total": external_count + trained_count,
            "external": external_count,
            "trained": trained_count,
            "active_monitoring": active_external
        },
        "monitoring": {
            "total_predictions_logged": total_predictions,
            "total_analyses_run": total_analyses
        },
        "compliance": {
            "critical_models": critical_models,
            "warning_models": warning_models,
            "compliant_models": trained_count - critical_models - warning_models
        }
    }