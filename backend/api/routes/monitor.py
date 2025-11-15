from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, field_validator
import pandas as pd
import numpy as np
import uuid
import io
from api.models.requests import PredictionLogRequest, BatchPredictionLogRequest
from api.models.responses import MonitoringStatsResponse
from api.models.database import get_db, PredictionLog
from core.src.logger import logging
from api.models.database import Model, ExternalModel
from datetime import timezone
from fastapi.responses import Response
from api.models.user import User
from core.auth.dependencies import get_current_active_user

router = APIRouter()


@router.post("/monitor/upload_csv")
async def upload_prediction_csv(
    model_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Upload CSV of production predictions for bias monitoring
    
    CSV Format (Required columns):
    - prediction: 0 or 1 (model output)
    - Sensitive attributes (as specified during model registration)
    
    CSV Format (Optional columns):
    - ground_truth: 0 or 1 (actual outcome)
    - prediction_proba: 0.0-1.0 (confidence score)
    - timestamp: Date of prediction
    - Any other feature columns
    
    Example CSV:
    prediction,race,gender,age,ground_truth,credit_score
    1,White,Male,35,1,720
    0,Black,Female,28,0,680
    1,Hispanic,Male,42,1,695
    ...
    
    Returns:
    - batch_id: Unique ID for this upload
    - predictions_logged: Number of predictions stored
    - validation_results: Any warnings/errors
    """
    
    try:
        logging.info(f"Uploading prediction CSV for model {model_id}")
        
        # Verify model exists (check both tables for backward compatibility)
        
        
        # Try external models first (BiasGuard 2.0)
        model = db.query(ExternalModel).filter(
                ExternalModel.model_id == model_id,
                ExternalModel.organization_id == current_user.organization_id
            ).first()

        if model:
                sensitive_cols = model.sensitive_attributes
                logging.info(f"Found external model: {model.model_name}")
        else:
                # Fallback to old models table
                old_model = db.query(Model).filter(
                    Model.model_id == model_id,
                    Model.organization_id == current_user.organization_id
                ).first()
                
                if old_model:
                    sensitive_cols = old_model.sensitive_columns
                    logging.info(f"Using legacy model format")
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model {model_id} not found or access denied"
                    )

        if not sensitive_cols:
                raise HTTPException(
                    status_code=400,
                    detail="Model has no sensitive attributes defined"
                )
        
        # Read CSV file
        try:
            contents = await file.read()
            df = pd.read_csv(io.BytesIO(contents))
            logging.info(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid CSV file: {str(e)}"
            )
        
        # Validate required columns
        validation_errors = []
        validation_warnings = []
        
        # Check for prediction column
        if 'prediction' not in df.columns:
            validation_errors.append("Missing required column: 'prediction'")
        
        # Check for sensitive attributes
        missing_sensitive = [col for col in sensitive_cols if col not in df.columns]
        if missing_sensitive:
            validation_errors.append(
                f"Missing sensitive attributes: {missing_sensitive}. "
                f"Required: {sensitive_cols}"
            )
        
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "CSV validation failed",
                    "errors": validation_errors
                }
            )
        
        # Check optional columns
        if 'ground_truth' not in df.columns:
            validation_warnings.append(
                "No 'ground_truth' column - analysis will use predictions only"
            )
        
        if 'prediction_proba' not in df.columns:
            validation_warnings.append(
                "No 'prediction_proba' column - confidence scores unavailable"
            )
        
        # Validate data types
        if not pd.api.types.is_integer_dtype(df['prediction']):
            try:
                df['prediction'] = df['prediction'].astype(int)
            except:
                validation_errors.append("'prediction' column must contain integers (0 or 1)")
        
        # Validate prediction values
        unique_preds = df['prediction'].unique()
        if not set(unique_preds).issubset({0, 1}):
            validation_errors.append(
                f"'prediction' must be 0 or 1. Found: {unique_preds}"
            )
        
        if validation_errors:
            raise HTTPException(status_code=400, detail={"errors": validation_errors})
        
        # Generate batch ID
        batch_id = f"batch_csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        logging.info(f"Processing batch: {batch_id}")
        
        # Prepare logs for bulk insert
        logs = []
        skipped = 0
        
        for idx, row in df.iterrows():
            try:
                # Extract sensitive attributes
                sensitive_attrs = {}
                for col in sensitive_cols:
                    value = row[col]
                    # Handle NaN values
                    if pd.isna(value):
                        logging.warning(f"Row {idx}: Missing value for {col}, skipping")
                        skipped += 1
                        continue
                    sensitive_attrs[col] = str(value) if not isinstance(value, (int, float)) else value
                
                # Skip if any sensitive attribute missing
                if len(sensitive_attrs) != len(sensitive_cols):
                    continue
                
                # Extract features (everything except prediction, ground_truth, sensitive attrs)
                feature_cols = [
                    col for col in df.columns 
                    if col not in ['prediction', 'ground_truth', 'prediction_proba', 'timestamp'] 
                    and col not in sensitive_cols
                ]
                features = {col: row[col] for col in feature_cols if not pd.isna(row[col])}
                
                # Handle timestamp
                if 'timestamp' in df.columns and not pd.isna(row['timestamp']):
                    try:
                        logged_at = pd.to_datetime(row['timestamp'])
                    except:
                        logged_at = datetime.now(timezone.utc)

                else:
                    logged_at = datetime.now(timezone.utc)
                
                # Create log entry
                log = PredictionLog(
                    log_id=f"log_{uuid.uuid4().hex[:16]}",
                    model_id=model_id,
                    prediction=int(row['prediction']),
                    prediction_proba=float(row['prediction_proba']) if 'prediction_proba' in df.columns and not pd.isna(row['prediction_proba']) else None,
                    ground_truth=int(row['ground_truth']) if 'ground_truth' in df.columns and not pd.isna(row['ground_truth']) else None,
                    features=features if features else None,
                    sensitive_attributes=sensitive_attrs,
                    organization_id=current_user.organization_id,
                    batch_id=batch_id,
                    data_source="csv",
                    logged_at=logged_at
                )
                
                logs.append(log)
                
            except Exception as e:
                logging.warning(f"Row {idx} skipped: {e}")
                skipped += 1
                continue
        
        if len(logs) == 0:
            raise HTTPException(
                status_code=400,
                detail="No valid predictions found in CSV"
            )
        
        # Bulk insert to database
        logging.info(f"Saving {len(logs)} prediction logs to database...")
        db.bulk_save_objects(logs)
        db.commit()
        
        logging.info(f"Batch {batch_id} uploaded successfully")
        
        # Calculate upload statistics
        predictions_array = np.array([log.prediction for log in logs])
        approval_rate = float(np.mean(predictions_array))
        
        # Get breakdown by sensitive attributes
        breakdown = {}
        for attr in sensitive_cols:
            attr_values = [log.sensitive_attributes.get(attr) for log in logs]
            unique_vals = list(set(attr_values))
            
            attr_breakdown = {}
            for val in unique_vals:
                mask = [log.sensitive_attributes.get(attr) == val for log in logs]
                val_preds = predictions_array[mask]
                attr_breakdown[str(val)] = {
                    "count": int(np.sum(mask)),
                    "approval_rate": float(np.mean(val_preds))
                }
            
            breakdown[attr] = attr_breakdown
        
        return {
            "status": "success",
            "batch_id": batch_id,
            "model_id": model_id,
            "predictions_logged": len(logs),
            "skipped_rows": skipped,
            "validation": {
                "errors": validation_errors,
                "warnings": validation_warnings
            },
            "statistics": {
                "total_predictions": len(logs),
                "overall_approval_rate": approval_rate,
                "has_ground_truth": any(log.ground_truth is not None for log in logs),
                "date_range": {
                    "earliest": min(log.logged_at for log in logs).isoformat(),
                    "latest": max(log.logged_at for log in logs).isoformat()
                },
                "breakdown_by_attribute": breakdown
            },
            "next_steps": {
                "analyze": f"POST /api/v1/analyze with model_id={model_id}",
                "view_dashboard": f"GET /dashboard/{model_id}",
                "generate_report": f"POST /api/v1/reports/generate/{model_id}"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"CSV upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# ========================================
# REAL-TIME LOGGING ENDPOINTS (Future Phase)
# ========================================

@router.post("/monitor/log")
async def log_single_prediction(
    request: PredictionLogRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Log a single production prediction in real-time
    
    Use this endpoint to stream predictions as they happen.
    For batch uploads, use /monitor/upload_csv instead.
    """
    
    try:
        model = db.query(ExternalModel).filter(
            ExternalModel.model_id == request.model_id,
            ExternalModel.organization_id == current_user.organization_id
        ).first()
        
        if not model:
            # Try legacy table
            old_model = db.query(Model).filter(Model.model_id == request.model_id, Model.organization_id == current_user.organization_id).first()
            if not old_model:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Model {request.model_id} not found"
                )
        
        # Create log entry
        log = PredictionLog(
            log_id=f"log_{uuid.uuid4().hex[:16]}",
            model_id=request.model_id,
            prediction=request.prediction,
            prediction_proba=request.prediction_proba,
            ground_truth=request.ground_truth,
            features=request.features,
            sensitive_attributes=request.sensitive_attributes,
            organization_id=current_user.organization_id,
            data_source="api",
            logged_at=datetime.now(timezone.utc)
        )
        
        db.add(log)
        db.commit()
        
        return {
            "status": "success",
            "log_id": log.log_id,
            "logged_at": log.logged_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to log prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/monitor/log_batch")
async def log_batch_predictions(
    request: BatchPredictionLogRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Log a batch of predictions via API (JSON)
    
    Use this for programmatic batch uploads.
    For CSV files, use /monitor/upload_csv instead.
    """
    
    try:
        
        model = db.query(ExternalModel).filter(
            ExternalModel.model_id == request.model_id,
            ExternalModel.organization_id == current_user.organization_id
        ).first()
        
        if not model:
            old_model = db.query(Model).filter(Model.model_id == request.model_id, Model.organization_id == current_user.organization_id).first()
            if not old_model:
                raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
        
        # Validate lengths match
        if request.prediction_probas and len(request.prediction_probas) != len(request.predictions):
            raise HTTPException(
                status_code=400,
                detail="prediction_probas length must match predictions"
            )
        
        if request.ground_truths and len(request.ground_truths) != len(request.predictions):
            raise HTTPException(
                status_code=400,
                detail="ground_truths length must match predictions"
            )
        
        # Generate batch ID
        batch_id = f"batch_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        # Create logs
        logs = []
        for i in range(len(request.predictions)):
            log = PredictionLog(
                log_id=f"log_{uuid.uuid4().hex[:16]}",
                model_id=request.model_id,
                prediction=request.predictions[i],
                prediction_proba=request.prediction_probas[i] if request.prediction_probas else None,
                ground_truth=request.ground_truths[i] if request.ground_truths else None,
                features=request.features[i] if request.features else None,
                sensitive_attributes=request.sensitive_attributes[i],
                organization_id=current_user.organization_id,
                batch_id=batch_id,
                data_source="api_batch",
                logged_at=datetime.now(timezone.utc)
            )
            logs.append(log)
        
        # Bulk save
        db.bulk_save_objects(logs)
        db.commit()
        
        logging.info(f"Batch {batch_id}: {len(logs)} predictions logged")
        
        return {
            "status": "success",
            "batch_id": batch_id,
            "predictions_logged": len(logs),
            "model_id": request.model_id,
            "logged_at": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Batch logging failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/monitor/stats/{model_id}")
async def get_monitoring_stats(
    model_id: str,
    days: int = 7,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> MonitoringStatsResponse:
    """
    Get monitoring statistics for a model
    
    Args:
        model_id: Model to get stats for
        days: Number of days to analyze (default: 7)
    
    Returns:
        Statistics about logged predictions
    """
    
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Query logs
        logs = db.query(PredictionLog).filter(
            PredictionLog.model_id == model_id,
            PredictionLog.organization_id == current_user.organization_id,
            PredictionLog.logged_at >= cutoff_date
        ).all()
        
        if not logs:
            return MonitoringStatsResponse(
                model_id=model_id,
                period_days=days,
                total_predictions=0,
                approval_rate=0.0,
                predictions_per_day=0.0,
                latest_prediction="N/A",
                has_ground_truth=False
            )
        
        # Calculate statistics
        predictions = [log.prediction for log in logs]
        approval_rate = sum(predictions) / len(predictions)
        
        return MonitoringStatsResponse(
            model_id=model_id,
            period_days=days,
            total_predictions=len(logs),
            approval_rate=approval_rate,
            predictions_per_day=len(logs) / days,
            latest_prediction=max(log.logged_at for log in logs).isoformat(),
            has_ground_truth=any(log.ground_truth is not None for log in logs)
        )
        
    except Exception as e:
        logging.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitor/batches/{model_id}")
async def list_batches(
    model_id: str,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    List all uploaded batches for a model
    
    Returns batch IDs, upload times, and sample counts
    """
    
    try:
        # Query distinct batches
        from sqlalchemy import func, distinct
        
        batches = db.query(
            PredictionLog.batch_id,
            func.count(PredictionLog.log_id).label('count'),
            func.min(PredictionLog.logged_at).label('uploaded_at'),
            PredictionLog.data_source
        ).filter(
            PredictionLog.model_id == model_id,
            PredictionLog.batch_id.isnot(None)
        ).group_by(
            PredictionLog.batch_id,
            PredictionLog.data_source
        ).order_by(
            func.min(PredictionLog.logged_at).desc()
        ).limit(limit).all()
        
        return {
            "model_id": model_id,
            "total_batches": len(batches),
            "batches": [
                {
                    "batch_id": batch.batch_id,
                    "predictions": batch.count,
                    "uploaded_at": batch.uploaded_at.isoformat(),
                    "source": batch.data_source
                }
                for batch in batches
            ]
        }
        
    except Exception as e:
        logging.error(f"Failed to list batches: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/monitor/batch/{batch_id}")
async def delete_batch(
    batch_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a specific batch of predictions
    
    Use this to clean up test data or incorrect uploads
    """
    
    try:
        # Delete all logs in batch
        deleted = db.query(PredictionLog).filter(
            PredictionLog.batch_id == batch_id
        ).delete()
        
        db.commit()
        
        if deleted == 0:
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
        
        logging.info(f"Deleted batch {batch_id}: {deleted} predictions removed")
        
        return {
            "status": "success",
            "batch_id": batch_id,
            "predictions_deleted": deleted
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to delete batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================================
# HELPER ENDPOINTS
# ========================================

@router.get("/monitor/download_template")
async def download_csv_template(
    sensitive_attributes: str = "race,gender,age"
):
    """
    Download a CSV template for prediction uploads
    
    Args:
        sensitive_attributes: Comma-separated list (e.g., "race,gender,age")
    
    Returns:
        CSV template file
    """
    
    try:
        attrs = [attr.strip() for attr in sensitive_attributes.split(',')]
        
        # Create template DataFrame
        template_data = {
            'prediction': [1, 0, 1],
            'prediction_proba': [0.85, 0.42, 0.78],
            'ground_truth': [1, 0, 1],
        }
        
        # Add sensitive attributes
        examples = {
            'race': ['White', 'Black', 'Hispanic'],
            'gender': ['Male', 'Female', 'Male'],
            'age': [35, 28, 42],
            'ethnicity': ['Non-Hispanic', 'Non-Hispanic', 'Hispanic'],
            'marital_status': ['Married', 'Single', 'Married']
        }
        
        for attr in attrs:
            if attr in examples:
                template_data[attr] = examples[attr]
            else:
                template_data[attr] = ['value1', 'value2', 'value3']
        
        # Add example features
        template_data['credit_score'] = [720, 680, 695]
        template_data['annual_income'] = [75000, 45000, 62000]
        
        df = pd.DataFrame(template_data)
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=biasguard_upload_template.csv"
            }
        )
        
    except Exception as e:
        logging.error(f"Failed to generate template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitor/health")
async def monitoring_health_check(db: Session = Depends(get_db)):
    """
    Health check for monitoring service
    
    Returns system stats and status
    """
    
    try:
        # Count total logs
        total_logs = db.query(PredictionLog).count()
        
        # Count logs from last 24 hours
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        recent_logs = db.query(PredictionLog).filter(
            PredictionLog.logged_at >= yesterday
        ).count()
        
        # Count unique models being monitored
        from sqlalchemy import func
        unique_models = db.query(func.count(func.distinct(PredictionLog.model_id))).scalar()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "statistics": {
                "total_predictions_logged": total_logs,
                "predictions_last_24h": recent_logs,
                "models_being_monitored": unique_models
            }
        }
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }