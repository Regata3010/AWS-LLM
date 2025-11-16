from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import Dict, Optional
from pydantic import BaseModel
import os
from datetime import datetime
from pathlib import Path
from api.models.requests import ReportGenerateRequest
from api.models.responses import ReportGenerateResponse
from api.models.database import get_db, ExternalModel, Model, PredictionLog 
from api.models import crud
from core.reports.report_generator import ComplianceReportGenerator
from core.src.logger import logging

router = APIRouter()

# Ensure reports directory exists
REPORTS_DIR = Path("/app/static/reportspdfs")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/reports/generate/{model_id}", response_model=ReportGenerateResponse)
async def generate_report(
    model_id: str,
    request: ReportGenerateRequest,
    db: Session = Depends(get_db)
):
    """
    Generate compliance report for a model
    Supports both legacy Model and new ExternalModel
    """
    try:
        logging.info(f"Generating {request.report_type} report for {model_id}")
        
        # Try ExternalModel first (BiasGuard 2.0)
        external_model = db.query(ExternalModel).filter(
            ExternalModel.model_id == model_id
        ).first()
        
        if external_model:
            logging.info(f"Found external model: {external_model.model_name}")
            
            # Convert ExternalModel to Model-like object for report generator
            # Count predictions for stats
            prediction_count = db.query(PredictionLog).filter(
                PredictionLog.model_id == model_id
            ).count()
            
            # Create compatible model object
            model = type('ModelProxy', (object,), {
                'model_id': external_model.model_id,
                'model_type': external_model.framework or external_model.model_type or "External Model",
                'task_type': 'binary',  # Default
                'dataset_name': 'Production Predictions',
                'target_column': 'prediction',
                'sensitive_columns': external_model.sensitive_attributes,
                'feature_count': 0,  # Unknown for external models
                'training_samples': 0,  # Not applicable
                'test_samples': prediction_count,  # Use prediction count
                'accuracy': 0.0,  # Calculate from ground truth if available
                'bias_status': 'unknown',  # Will get from bias_analysis
                'mlflow_run_id': None,
                'created_at': external_model.created_at,
                'updated_at': external_model.updated_at
            })()
            
            # Try to calculate accuracy if ground truth available
            logs_with_truth = db.query(PredictionLog).filter(
                PredictionLog.model_id == model_id,
                PredictionLog.ground_truth.isnot(None)
            ).all()
            
            if logs_with_truth:
                predictions = [log.prediction for log in logs_with_truth]
                ground_truths = [log.ground_truth for log in logs_with_truth]
                accuracy = sum(p == g for p, g in zip(predictions, ground_truths)) / len(predictions)
                model.accuracy = accuracy
                model.training_samples = len(logs_with_truth)
                logging.info(f"   Calculated accuracy from {len(logs_with_truth)} samples: {accuracy:.2%}")
            
        else:
            # Fallback to old Model table
            model = crud.get_model(db, model_id)
            if not model:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            logging.info(f"Found legacy model: {model.model_type}")
        
        # Fetch latest bias analysis (works for both model types)
        bias_analysis = crud.get_latest_bias_analysis(db, model_id)
        if not bias_analysis:
            raise HTTPException(
                status_code=404, 
                detail=f"No bias analysis found for model {model_id}. Run analysis first."
            )
        
        # Update model bias_status from analysis
        if hasattr(model, '__dict__'):
            model.bias_status = bias_analysis.bias_status
        
        # Initialize report generator
        generator = ComplianceReportGenerator()
        
        # Generate report
        report_data = await generator.generate_report(
            model=model,
            bias_analysis=bias_analysis,
            report_type=request.report_type,
            include_recommendations=request.include_recommendations,
            include_technical=request.include_technical_details
        )
        
        # Generate unique report ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_id = f"report_{model_id}_{timestamp}"
        
        # Save PDF using ReportLab
        output_path = REPORTS_DIR / f"{report_id}.pdf"
        generator.create_pdf(report_data["report_data"], output_path)
        file_extension = "pdf"
        
        # Get file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        logging.info(f"Report generated: {output_path} ({file_size_mb:.2f}MB)")
        
        return ReportGenerateResponse(
            report_id=report_id,
            download_url=f"/api/v1/reports/download/{report_id}.{file_extension}",
            generated_at=datetime.now().isoformat(),
            model_id=model_id,
            report_type=request.report_type,
            file_size_mb=round(file_size_mb, 2),
            llm_summary=report_data.get("executive_summary")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@router.get("/reports/download/{filename}")
async def download_report(filename: str):
    """
    Download generated report
    
    Returns PDF or HTML file for download
    """
    file_path = REPORTS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Determine media type
    media_type = "application/pdf" if filename.endswith(".pdf") else "text/html"
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename,
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )

@router.get("/reports/list")
async def list_reports():
    """List all generated reports"""
    try:
        reports = []
        for file_path in REPORTS_DIR.glob("report_*.pdf"):
            stat = file_path.stat()
            reports.append({
                "filename": file_path.name,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "download_url": f"/reports/download/{file_path.name}"
            })
        
        return {
            "total": len(reports),
            "reports": sorted(reports, key=lambda x: x["created_at"], reverse=True)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/reports/delete/{filename}")
async def delete_report(filename: str):
    """Delete a generated report"""
    file_path = REPORTS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    try:
        file_path.unlink()
        return {"status": "success", "message": f"Report {filename} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))