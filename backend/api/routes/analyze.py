from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import numpy as np
import uuid
import mlflow
import json
from datetime import timezone
from api.models.database import get_db, ExternalModel, PredictionLog, Model
from api.models import schemas, crud
from core.bias_detector.detector import generate_report_fairness
from core.bias_detector.intersectionality import IntersectionalityAnalyzer
from core.src.logger import logging
import mlflow
import json
import time
from api.models.requests import AnalyzeRequest
from api.models.user import User
from core.auth.dependencies import get_current_active_user
import os


mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns"))

try:
    experiment = mlflow.get_experiment_by_name("updated_bias_monitoring")
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            "updated_bias_monitoring",
            tags={
                "purpose": "production_monitoring",
                "platform": "biasguard_2.0"
            }
        )
        logging.info(f"Created MLflow experiment 'updated_bias_monitoring'")
    else:
        logging.info(f"Using experiment updated_bias_monitoring'")
except Exception as e:
    logging.warning(f"MLflow experiment setup failed: {e}")


router = APIRouter()

def clean_for_json(data):
    """Convert non-JSON-serializable types to JSON-compatible types"""
    if isinstance(data, dict):
        return {k: clean_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_for_json(v) for v in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, bool):
        return bool(data)
    elif isinstance(data, (int, np.integer)):
        return int(data)
    elif isinstance(data, (float, np.floating)):
        val = float(data)
        if val == float('inf'):
            return "inf"
        elif val == float('-inf'):
            return "-inf"
        else:
            return val
    elif data == float('inf'):
        return "inf"
    elif data == float('-inf'):
        return "-inf"
    else:
        return data


@router.post("/analyze")
async def analyze_logged_predictions(
    request: AnalyzeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Analyze bias in production predictions (BiasGuard 2.0 monitoring mode)
    
    This analyzes predictions that were uploaded via /monitor/upload_csv
    instead of predictions from trained models.
    """
    
    try:
        logging.info(f"Analyzing logged predictions for model {request.model_id}")
        
        # Start MLflow run for this analysis
        mlflow.set_experiment("updated_bias_monitoring")
        
        with mlflow.start_run(run_name=f"monitor_{request.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            
            # Tag the run
            mlflow.set_tag("model_id", request.model_id)
            mlflow.set_tag("analysis_type", "production_monitoring")
            mlflow.set_tag("biasguard_version", "2.0")
            
            # Log analysis parameters
            mlflow.log_param("period_days", request.period_days)
            mlflow.log_param("min_samples", request.min_samples)
        
        # Get model (try ExternalModel first, fallback to Model)
            external_model = db.query(ExternalModel).filter(
                    ExternalModel.model_id == request.model_id,
                    ExternalModel.organization_id == current_user.organization_id
                ).first()
                
            if external_model:
                sensitive_cols = external_model.sensitive_attributes
                mlflow.set_tag("model_name", external_model.model_name)
                mlflow.set_tag("model_source", "external")
                mlflow.log_param("framework", external_model.framework)
                logging.info(f"Found external model: {external_model.model_name}")
            else:
                # Fallback to old Model table
                old_model = db.query(Model).filter(Model.model_id == request.model_id).first()
                if old_model:
                    sensitive_cols = old_model.sensitive_columns
                    mlflow.set_tag("model_name", old_model.model_type)
                    mlflow.set_tag("model_source", "trained_internal")
                    logging.info(f"Using legacy model format")
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model {request.model_id} not found"
                    )
            
            # Query prediction logs
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=request.period_days)
            
            logs = db.query(PredictionLog).filter(
                    PredictionLog.model_id == request.model_id,
                    PredictionLog.organization_id == current_user.organization_id,
                    PredictionLog.logged_at >= cutoff_date
                ).order_by(PredictionLog.logged_at.asc()).all()
                
            if len(logs) < request.min_samples:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient data: {len(logs)} predictions found, minimum required: {request.min_samples}"
                )
            
            # Log dataset metrics to MLflow
            mlflow.log_metric("samples_analyzed", len(logs))
            mlflow.log_param("date_range_start", logs[0].logged_at.isoformat())
            mlflow.log_param("date_range_end", logs[-1].logged_at.isoformat())
            
            logging.info(f"Analyzing {len(logs)} predictions from {logs[0].logged_at} to {logs[-1].logged_at}")
            
            # Extract predictions and sensitive attributes
            predictions = np.array([log.prediction for log in logs])
            
            # Extract ground truth if available
            ground_truths = [log.ground_truth for log in logs if log.ground_truth is not None]
            has_ground_truth = len(ground_truths) > 0
            
            if has_ground_truth:
                ground_truths = np.array(ground_truths)
                logging.info(f"Ground truth available for {len(ground_truths)} samples")
            else:
                logging.info("No ground truth - using predictions as proxy")
                ground_truths = predictions
            
            # Build sensitive data dictionary
            sensitive_data = {}
            for attr in sensitive_cols:
                values = [log.sensitive_attributes.get(attr) for log in logs]
                sensitive_data[attr] = np.array(values)
                logging.info(f"   {attr}: {len(set(values))} unique values")
            
            # Run fairness analysis for each sensitive attribute
            all_fairness_results = {}
            
            for attr_name, attr_values in sensitive_data.items():
                logging.info(f"Running fairness analysis for: {attr_name}")
                
                fairness_report = generate_report_fairness(
                    y_true=ground_truths,
                    y_pred=predictions,
                    sensitive_attr=attr_values
                )
                
                all_fairness_results[attr_name] = {
                    "statistical_parity": fairness_report['Statistical_Parity'],
                    "equal_opportunity": fairness_report['Equal_Opportunity'],
                    "disparate_impact": fairness_report['Disparate_Impact_Ratio'],
                    "average_odds": fairness_report['Average_Odds_Difference'],
                    "fpr_parity": fairness_report['False_Positive_Rate_Parity'],
                    "predictive_parity": fairness_report['Predictive_Parity'],
                    "treatment_equality": fairness_report['Treatment_Equality']
                }
                
                # Log key metrics
                di = fairness_report['Disparate_Impact_Ratio'].get('ratio', 1)
                sp = fairness_report['Statistical_Parity'].get('statistical_parity_diff', 0)
                logging.info(f"   DI: {di:.4f}, SP: {sp:.4f}")
            
            # Run intersectionality analysis
            intersectionality_results = None
            if len(sensitive_data) >= 2:
                logging.info(f"Running intersectionality analysis...")
                
                try:
                    analyzer = IntersectionalityAnalyzer(
                        min_group_size=30,
                        di_threshold=0.8,
                        max_combinations=20
                    )
                    
                    intersectionality_results = analyzer.analyze(
                        y_test_array=predictions,
                        y_pred_array=predictions,
                        sensitive_dict=sensitive_data
                    )
                    
                    if intersectionality_results and "summary" in intersectionality_results:
                        logging.info(
                            f"   Groups analyzed: {intersectionality_results['summary']['total_groups_analyzed']}, "
                            f"At risk: {intersectionality_results['summary']['groups_below_threshold']}"
                        )
                        
                except Exception as e:
                    logging.error(f"Intersectionality failed: {e}")
                    intersectionality_results = {"error": str(e)}
            
            # Assess compliance
            compliance_status = _assess_compliance(all_fairness_results)
            bias_status = _determine_bias_status(compliance_status)
            
            # Override if intersectional bias detected
            if intersectionality_results and "summary" in intersectionality_results:
                if intersectionality_results["summary"]["groups_below_threshold"] > 0:
                    if "COMPLIANT" in compliance_status:
                        compliance_status = "WARNING - Intersectional bias detected"
                        bias_status = "warning"
            
            # Generate recommendations
            recommendations = _generate_recommendations(all_fairness_results, intersectionality_results)
            
            # Check alerts
            alerts = _check_alerts(all_fairness_results, sensitive_cols)
            
            # CLEAN DATA before saving to database (moved earlier)
            all_fairness_results_clean = clean_for_json(all_fairness_results)
            intersectionality_results_clean = clean_for_json(intersectionality_results) if intersectionality_results else None
            recommendations_clean = [str(r) for r in recommendations]  # Ensure strings
            
            # Save analysis to database
            analysis_id = f"analysis_{uuid.uuid4().hex[:12]}"
            
            analysis_data = schemas.BiasAnalysisCreate(
                analysis_id=analysis_id,
                model_id=request.model_id,
                compliance_status=compliance_status,
                bias_status=bias_status,
                fairness_metrics=all_fairness_results_clean,  # Use cleaned version
                aif360_metrics=None,
                organization_id=current_user.organization_id,
                recommendations=recommendations_clean,  # Use cleaned version
                mlflow_run_id="monitoring_analysis"
            )
            
            crud.create_bias_analysis(db, analysis_data)
            logging.info(f"Analysis saved: {analysis_id}")
            
            # Update model bias status (try both tables)
            if external_model:
                # Can't update ExternalModel bias_status (it doesn't have that field)
                pass
            else:
                crud.update_model_bias_status(db, request.model_id, bias_status)
            
            # Send alerts if necessary
            if alerts:
                logging.warning(f"{len(alerts)} alerts triggered!")
                for alert in alerts:
                    logging.warning(f"   {alert['type'].upper()}: {alert['message']}")
    
        return {
            "analysis_id": analysis_id,
            'mlflow_run_id': run.info.run_id,
            "model_id": request.model_id,
            "period": {
                "start": logs[0].logged_at.isoformat(),
                "end": logs[-1].logged_at.isoformat(),
                "days": request.period_days,
                "samples": len(logs)
            },
            "fairness_metrics": all_fairness_results_clean,  
            "intersectionality": intersectionality_results_clean,  
            "compliance_status": compliance_status,
            "bias_status": bias_status,
            "recommendations": recommendations_clean,
            "alerts": alerts,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "has_ground_truth": has_ground_truth
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _assess_compliance(fairness_results: Dict) -> str:
    """Assess compliance based on fairness metrics"""
    
    violations = []
    warnings = []
    
    for attr, metrics in fairness_results.items():
        # Check disparate impact
        di = metrics['disparate_impact'].get('ratio', 1)
        if di not in ['inf', float('inf')]:
            if di < 0.8 or di > 1.25:
                violations.append(f"{attr}_DI={di:.3f}")
            elif di < 0.85 or di > 1.20:
                warnings.append(f"{attr}_DI={di:.3f}")
        
        # Check statistical parity
        sp = abs(metrics['statistical_parity'].get('statistical_parity_diff', 0))
        if sp > 0.2:
            violations.append(f"{attr}_SP={sp:.3f}")
        elif sp > 0.1:
            warnings.append(f"{attr}_SP={sp:.3f}")
    
    if violations:
        return f"NON-COMPLIANT - {len(violations)} violations detected"
    elif warnings:
        return f"WARNING - {len(warnings)} metrics at risk"
    else:
        return "COMPLIANT - All metrics within thresholds"

def _determine_bias_status(compliance_status: str) -> str:
    """Convert compliance status to bias status"""
    
    if "NON-COMPLIANT" in compliance_status:
        return "critical"
    elif "WARNING" in compliance_status:
        return "warning"
    else:
        return "compliant"

def _generate_recommendations(fairness_results: Dict, intersectionality: Optional[Dict]) -> List[str]:
    """Generate actionable recommendations"""
    
    recommendations = []
    
    for attr, metrics in fairness_results.items():
        di = metrics['disparate_impact'].get('ratio', 1)
        
        if di not in ['inf', float('inf')]:
            if di < 0.5:
                recommendations.append(
                    f"[CRITICAL] Severe bias for {attr}: DI={di:.3f}. "
                    f"Immediate model review required. Consider retraining with fairness constraints."
                )
            elif di < 0.7:
                recommendations.append(
                    f"[HIGH] Significant bias for {attr}: DI={di:.3f}. "
                    f"Apply bias mitigation or adjust decision thresholds."
                )
            elif di < 0.8:
                recommendations.append(
                    f"[MODERATE] Bias detected for {attr}: DI={di:.3f} below 0.80 threshold. "
                    f"Monitor closely and consider mitigation."
                )
        
        # Statistical parity recommendations
        sp = abs(metrics['statistical_parity'].get('statistical_parity_diff', 0))
        if sp > 0.15:
            recommendations.append(
                f"[HIGH] Large statistical parity gap for {attr}: {sp:.3f}. "
                f"Review model features for proxy discrimination."
            )
    
    # Intersectionality recommendations
    if intersectionality and "summary" in intersectionality:
        if intersectionality["summary"]["groups_below_threshold"] > 0:
            recommendations.append(
                f"[HIGH] {intersectionality['summary']['groups_below_threshold']} intersectional groups "
                f"below threshold. Review compounding discrimination effects."
            )
            
            # Add specific worst group
            if intersectionality.get("worst_disparities"):
                worst = intersectionality["worst_disparities"][0]
                recommendations.append(
                    f"[CRITICAL] Worst group: {worst['group']} (DI={worst['disparate_impact']:.2f}). "
                    f"Conduct targeted bias audit."
                )
    
    if not recommendations:
        recommendations.append("Model shows acceptable fairness across all metrics. Continue monitoring.")
    
    return recommendations[:8]  # Top 8 recommendations

def _check_alerts(fairness_results: Dict, sensitive_cols: List[str]) -> List[Dict]:
    """Check if alerts should be triggered"""
    
    alerts = []
    
    for attr, metrics in fairness_results.items():
        # Disparate Impact alerts
        di = metrics['disparate_impact'].get('ratio', 1)
        if di not in ['inf', float('inf')]:
            if di < 0.8:
                alerts.append({
                    "type": "critical",
                    "metric": "disparate_impact",
                    "attribute": attr,
                    "value": di,
                    "threshold": 0.8,
                    "message": f"Disparate impact for {attr} below 0.80 threshold (CFPB violation)",
                    "regulation": "ECOA"
                })
            elif di > 1.25:
                alerts.append({
                    "type": "warning",
                    "metric": "disparate_impact",
                    "attribute": attr,
                    "value": di,
                    "threshold": 1.25,
                    "message": f"Disparate impact for {attr} above 1.25 threshold (reverse discrimination concern)"
                })
        
        # Statistical Parity alerts
        sp = abs(metrics['statistical_parity'].get('statistical_parity_diff', 0))
        if sp > 0.2:
            alerts.append({
                "type": "critical",
                "metric": "statistical_parity",
                "attribute": attr,
                "value": sp,
                "threshold": 0.2,
                "message": f"Statistical parity violation for {attr}: {sp:.3f}",
                "regulation": "Title VII"
            })
        elif sp > 0.1:
            alerts.append({
                "type": "warning",
                "metric": "statistical_parity",
                "attribute": attr,
                "value": sp,
                "threshold": 0.1,
                "message": f"Statistical parity approaching threshold for {attr}: {sp:.3f}"
            })
    
    return alerts