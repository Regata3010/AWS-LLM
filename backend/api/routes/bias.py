from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from api.models.requests import BiasDetectionRequest, MitigationRequest
from api.models.responses import BiasDetectionResponse, MitigationResponse
from api.routes.training import get_model_data, calculate_fairness_weights, store_model_data
from api.routes.upload import get_file_from_s3
from core.src.logger import logging
from core.bias_detector.detector import generate_report_fairness  
from core.bias_detector.aif360_wrapper import audit_with_aif360  
from core.bias_detector.mitigation import apply_reweighing, apply_reject_option 
from core.bias_detector.preprocessor import preprocess_data  
import mlflow
import json
from datetime import datetime 
from sklearn.metrics import accuracy_score
import uuid
import time
from contextlib import contextmanager
import os
from api.models.database import init_db, get_db
from api.models import schemas, crud    
from sqlalchemy.orm import Session
from sklearn.preprocessing import LabelEncoder
from api.models.database import SessionLocal
from core.bias_detector.mitigation_validator import MitigationValidator
from core.bias_detector.intersectionality import IntersectionalityAnalyzer
from deprecated import deprecated

# mlflow.set_tracking_uri("file:///Users/AWS-LLM/mlruns") #contadiction in the line below. i think this was done for keeping runs just check once
# ARTIFACT_ROOT = "/Users/AWS-LLM/backend/mlruns1"
# os.environ['MLFLOW_ARTIFACT_ROOT'] = ARTIFACT_ROOT
# os.makedirs(ARTIFACT_ROOT, exist_ok=True)


# try:
#     experiment = mlflow.get_experiment_by_name("bias_detection")
#     if experiment is None:
#         experiment_id = mlflow.create_experiment("bias_detection",artifact_location=f"file://{ARTIFACT_ROOT}/bias_detection")
#         logging.info(f"Created MLflow experiment 'bias_detection' with ID: {experiment_id}")
#     else:
#         logging.info(f"Using existing experiment 'bias_detection' with ID: {experiment.experiment_id}")
# except Exception as e:
#     logging.warning(f"Could not setup MLflow experiment: {e}")

@contextmanager
def safe_mlflow_run(run_name: str):
    """Ensures clean MLflow run management"""
    
    if mlflow.active_run():
        mlflow.end_run()
    
    run = mlflow.start_run(run_name=run_name)
    try:
        yield run
    finally:
        time.sleep(0.2)
        mlflow.end_run()

router = APIRouter()

def safe_float(value):
    """Convert to JSON-safe float"""
    if value == float('inf'):
        return 999.0
    elif value == float('-inf'):
        return -999.0
    elif isinstance(value, (int, float)):
        return float(value)
    else:
        return 0.0

def clean_for_json(data):
    """Convert numpy types and infinity to JSON-compatible types"""
    if isinstance(data, dict):
        return {k: clean_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_for_json(v) for v in data]
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, float):
        if data == float('inf'):
            return "inf"
        elif data == float('-inf'):
            return "-inf"
        else:
            return data
    elif data == "inf" or data == "-inf":
        return data  # Already a string
    else:
        return data
    
@router.post("/bias", response_model=BiasDetectionResponse, include_in_schema=False)
@deprecated(reason="Use BiasGuard 2.0 monitoring workflow instead")
async def detect_bias(request: BiasDetectionRequest, db: Session = Depends(get_db)):
    """
    Comprehensive bias detection using your existing metrics
    """
    try:
        # Get model data from training
        start_time = time.time()
        mlflow.set_experiment("bias_detection")
        model_data = get_model_data(request.model_id,db)
        
        #added to check model data
            # if not model_data or "y_test" not in model_data:
            #     raise HTTPException(
            #         status_code=404, 
            #         detail=f"Model {request.model_id} not found or incomplete"
            #     )
        
        parent_run_id = model_data.get("mlflow_run_id")
        
        with safe_mlflow_run(run_name=f"bias_detection_{request.model_id}") as run:
            if parent_run_id:
                mlflow.set_tag("parent_run_id", parent_run_id)
                
            mlflow.set_tag("analysis_type", "bias_detection")
            mlflow.log_param("model_id", request.model_id)
                
        
            y_test = model_data["y_test"]
            y_pred = model_data["y_pred"]
            sensitive_columns = model_data["sensitive_columns"]
            task_type = model_data["task_type"]
            
            # Handle sensitive data - works with your preprocessor output
            if "sensitive_data" in model_data:
                sensitive_data = model_data["sensitive_data"]
            else:
                # Handle legacy format from your preprocessor
                s_test = model_data["s_test"]
                if isinstance(sensitive_columns, list) and len(sensitive_columns) > 1:
                    # Multiple columns - s_test is 2D
                    sensitive_data = {}
                    for i, col in enumerate(sensitive_columns):
                        sensitive_data[col] = s_test[:, i]
                else:
                    # Single column
                    col_name = sensitive_columns[0] if isinstance(sensitive_columns, list) else sensitive_columns
                    sensitive_data = {col_name: s_test}
            logging.info(f"Sensitive columns for bias detection: {list(sensitive_data.keys())}")
            
            # Run your comprehensive fairness metrics for each sensitive attribute
            all_fairness_results = {}
            
            for col_name, col_values in sensitive_data.items():
                logging.info(f"Running fairness analysis for: {col_name}")
                
                # Use your existing detector - it returns all 7 metrics
                fairness_report = generate_report_fairness(y_test, y_pred, col_values)
                
                di_ratio = fairness_report['Disparate_Impact_Ratio'].get('ratio', 0)
                if di_ratio not in ['inf', float('inf'), float('-inf')]:
                    mlflow.log_metric(f"{col_name}_disparate_impact", float(di_ratio))
                else:
                    mlflow.log_metric(f"{col_name}_disparate_impact", 999.0)
                
                mlflow.log_metric(f"{col_name}_statistical_parity_diff", 
                                fairness_report['Statistical_Parity'].get('statistical_parity_diff', 0))
                mlflow.log_metric(f"{col_name}_equal_opportunity_diff", 
                                fairness_report['Equal_Opportunity'].get('difference', 0))
                mlflow.log_metric(f"{col_name}_average_odds_diff", 
                                fairness_report['Average_Odds_Difference'].get('average_odds_difference', 0))
                
                # Extract key information for response
                all_fairness_results[col_name] = {
                    "statistical_parity": fairness_report['Statistical_Parity'],
                    "equal_opportunity": fairness_report['Equal_Opportunity'],
                    "disparate_impact": fairness_report['Disparate_Impact_Ratio'],
                    "average_odds": fairness_report['Average_Odds_Difference'],
                    "fpr_parity": fairness_report['False_Positive_Rate_Parity'],
                    "predictive_parity": fairness_report['Predictive_Parity'],
                    "treatment_equality": fairness_report['Treatment_Equality']
                }
            
            #adding this intersectionality block
            intersectionality_results = None
            
            # Only run if we have 2+ sensitive attributes
            if len(sensitive_data) >= 2:
                logging.info(f"Running intersectionality on {len(sensitive_data)} attributes")
                
                try:
                    from core.bias_detector.intersectionality import IntersectionalityAnalyzer
                    
                    analyzer = IntersectionalityAnalyzer(
                        min_group_size=30,
                        di_threshold=0.8,
                        max_combinations=20
                    )
                    
                    # Call analyzer
                    intersect_result = analyzer.analyze(y_test, y_pred, sensitive_data)
                    
                    # Only update if successful
                    if intersect_result and "error" not in intersect_result:
                        intersectionality_results = intersect_result
                        
                        logging.info(
                            f"Intersectionality: {intersect_result['summary']['total_groups_analyzed']} groups, "
                            f"{intersect_result['summary']['groups_below_threshold']} at risk"
                        )
                        
                        # Log to MLflow
                        mlflow.log_metric("intersectional_groups_analyzed", intersect_result["summary"]["total_groups_analyzed"])
                        mlflow.log_metric("intersectional_groups_at_risk", intersect_result["summary"]["groups_below_threshold"])
                        
                        if intersect_result["summary"]["min_disparity"] != 999.0:
                            mlflow.log_metric("worst_intersectional_di", intersect_result["summary"]["min_disparity"])
                        
                        # Save artifact
                        intersect_path = f"/tmp/intersectionality_{request.model_id}.json"
                        with open(intersect_path, 'w') as f:
                            json.dump(intersect_result, f, indent=2, default=str)
                        mlflow.log_artifact(intersect_path)
                        time.sleep(0.1)
                    else:
                        logging.warning(f"Intersectionality returned error: {intersect_result.get('error')}")
                    
                except Exception as err:
                    logging.error(f"Intersectionality exception: {err}", exc_info=True)
                    intersectionality_results["recommendations"] = [f"Analysis crashed: {str(err)[:100]}"]
            else:
                intersectionality_results["recommendations"] = ["Need 2+ sensitive attributes"]
                    
    
            
            # Run AIF360 analysis for binary classification
            aif360_results = None
            if task_type == "binary" and len(sensitive_columns) > 0:
                try:
                    # Create dataframe for AIF360
                    X_test = model_data["X_test"]
                    df_for_aif = pd.DataFrame(X_test)
                    
                    # Add target and first sensitive column
                    df_for_aif['target'] = y_test
                    first_sensitive = list(sensitive_data.values())[0]
                    df_for_aif['sensitive'] = first_sensitive
                    
                    # Use your AIF360 wrapper
                    aif360_results = audit_with_aif360(
                        df_for_aif,
                        'target',
                        'sensitive'
                    )
                    logging.info(f"AIF360 results: {aif360_results}")
                    
                    if aif360_results and isinstance(aif360_results, dict):
                            # Log each AIF360 metric
                            for metric_name, metric_value in aif360_results.items():
                                if metric_name != "error" and isinstance(metric_value, (int, float)):
                                    # Clean the metric value
                                    if metric_value == float('inf'):
                                        mlflow.log_metric(f"aif360_{metric_name}", 999.0)
                                    elif metric_value == float('-inf'):
                                        mlflow.log_metric(f"aif360_{metric_name}", -999.0)
                                    else:
                                        mlflow.log_metric(f"aif360_{metric_name}", float(metric_value))
                                elif isinstance(metric_value, dict):
                                   
                                    for sub_metric, sub_value in metric_value.items():
                                        if isinstance(sub_value, (int, float)):
                                            mlflow.log_metric(f"aif360_{metric_name}_{sub_metric}", float(sub_value))
                    
                except Exception as e:
                    logging.warning(f"AIF360 analysis failed: {e}")
                    aif360_results = {"error": str(e)}
                    mlflow.log_param("aif360_error", str(e))
            
            # Generate overall assessment
            compliance_status = assess_compliance(all_fairness_results)
            mlflow.set_tag("compliance_status", compliance_status)
            bias_detected = "NON_COMPLIANT" in compliance_status or "WARNING" in compliance_status
            mlflow.log_metric("bias_detected", 1 if bias_detected else 0)
            
            if "NON_COMPLIANT" in compliance_status:
                bias_status = "critical"
            elif "WARNING" in compliance_status:
                bias_status = "warning"
            else:
                bias_status = "compliant"
                
            if intersectionality_results and "error" not in intersectionality_results:
                if intersectionality_results["summary"]["groups_below_threshold"] > 0:
                    critical_intersectional = any(
                        d.get("severity") == "CRITICAL" 
                        for d in intersectionality_results.get("worst_disparities", [])
                    )
                    
                    if critical_intersectional and "COMPLIANT" in compliance_status:
                        compliance_status = "WARNING - Intersectional bias detected"
                        bias_status = "warning"
                        mlflow.set_tag("intersectional_override", "true")
            
            recommendations = generate_recommendations(all_fairness_results, model_data.get("fairness_applied", False))
            if intersectionality_results and "recommendations" in intersectionality_results:
                recommendations.extend(intersectionality_results["recommendations"])
            
            mlflow.log_text("\n".join(recommendations), "recommendations.txt")
            
            
            all_fairness_results = clean_for_json(all_fairness_results)
            if aif360_results:
                aif360_results = clean_for_json(aif360_results)
            
            report_path = f"/tmp/fairness_report_{request.model_id}.json"
            with open(report_path, 'w') as f:
                json.dump(all_fairness_results, f, indent=2, default=str)
            mlflow.log_artifact(report_path)
            time.sleep(0.2)
            logging.info(f"Bias Detection MLflow run: {run.info.run_id}")
            
            # if intersectionality_results and "error" not in intersectionality_results:
            #     intersect_path = f"/tmp/intersectionality_report_{request.model_id}.json"
            #     with open(intersect_path, 'w') as f:
            #         json.dump(intersectionality_results, f, indent=2, default=str)
            #     mlflow.log_artifact(intersect_path)
            #     time.sleep(0.2)
            
            latency = time.time() - start_time
            
            if latency > 0.2:
                logging.warning(f"Bias detection exceeded 100ms SLA: {latency:.3f}s")
                mlflow.set_tag("sla_violation", "true")
                
            analysis_data = schemas.BiasAnalysisCreate(
            analysis_id=f"analysis_{uuid.uuid4().hex[:8]}",
            model_id=request.model_id,
            compliance_status=compliance_status,
            bias_status=bias_status,
            fairness_metrics=all_fairness_results,
            aif360_metrics=aif360_results,
            recommendations=recommendations,
            mlflow_run_id=run.info.run_id
            )
            
            crud.create_bias_analysis(db, analysis_data)
            logging.info(f"Saved bias analysis for model {request.model_id}")
            
            
            crud.update_model_bias_status(db, request.model_id, bias_status)
            logging.info(f"Updated model {request.model_id} bias status: {bias_status}")
                
            return BiasDetectionResponse(
                model_id=request.model_id,
                mlflow_run_id=run.info.run_id,
                fairness_metrics=all_fairness_results,
                aif360_metrics=aif360_results,
                compliance_status=compliance_status,
                recommendations=recommendations,
                latency=latency,
                task_type=task_type,
                fairness_applied=model_data.get("fairness_applied", False),
                intersectionality_analysis=intersectionality_results 
            )
            
    except Exception as e:
        logging.error(f"Bias detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bias/mitigate", response_model=MitigationResponse, include_in_schema=False)
@deprecated(reason="Use BiasGuard 2.0 monitoring workflow instead")
async def mitigate_bias(request: MitigationRequest, db: Session=Depends(get_db)):
    """
    Apply bias mitigation through retraining or threshold optimization
    """
  
    try:
        # Get original model data
        mlflow.set_experiment("bias_detection")
        model_data = get_model_data(request.model_id, db)
        if not model_data or "y_test" not in model_data:
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found or incomplete")
        
        parent_run_id = model_data.get("mlflow_run_id")
        
        with safe_mlflow_run(run_name=f"mitigation_{request.model_id}_{request.mitigation_strategy}") as run:
            #check logging
            mlflow.set_tag("mitigation_type", request.mitigation_strategy)
            mlflow.set_tag("parent_model_id", request.model_id)
            if parent_run_id:
                mlflow.set_tag("parent_run_id", parent_run_id)
            
            mlflow.log_param("original_model_id", request.model_id)
            mlflow.log_param("mitigation_strategy", request.mitigation_strategy)
            mlflow.log_param("max_accuracy_loss", request.max_accuracy_loss)
        
            # Get current bias metrics
            y_test = model_data["y_test"]
            y_pred = model_data["y_pred"]
            
            s_test = model_data["s_test"]
            
            # Use first sensitive column for mitigation
            s_test_1d = s_test[:, 0] if s_test.ndim > 1 else s_test
            
            # Calculate original bias
            original_bias = generate_report_fairness(y_test, y_pred, s_test_1d)
            original_accuracy = accuracy_score(y_test, y_pred)
            
            mlflow.log_metric("original_accuracy", original_accuracy)
            
            original_di = original_bias['Disparate_Impact_Ratio'].get('ratio', 1)
            if original_di != 'inf' and original_di != float('inf'):
                mlflow.log_metric("original_disparate_impact", float(original_di))
            else:
                mlflow.log_metric("original_disparate_impact", 999.0)
            
            # Log other original metrics
            mlflow.log_metric("original_statistical_parity_diff", 
                            float(original_bias['Statistical_Parity'].get('statistical_parity_diff', 0)))
            mlflow.log_metric("original_equal_opportunity_diff",
                            float(original_bias['Equal_Opportunity'].get('difference', 0)))
            
            # Check if mitigation is needed
            di = original_bias['Disparate_Impact_Ratio'].get('ratio', 1)
            if di != 'inf' and 0.8 <= di <= 1.25:
                return MitigationResponse(
                    status="not_needed",
                    original_model_id=request.model_id,
                    new_model_id=request.model_id,
                    mlflow_run_id =run.info.run_id, 
                    strategy_applied="none",
                    original_metrics={
                        "accuracy": original_accuracy,
                        "disparate_impact": clean_for_json(di),
                        "statistical_parity_diff": original_bias['Statistical_Parity'].get('statistical_parity_diff', 0)
                    },
                    new_metrics={
                        "accuracy": original_accuracy,
                        "disparate_impact": clean_for_json(di),
                        "statistical_parity_diff": original_bias['Statistical_Parity'].get('statistical_parity_diff', 0)
                    },
                    bias_improvement={},
                    accuracy_impact=0.0,
                    message="Model already meets fairness criteria (0.8 <= DI <= 1.25)"
                )
            
            # Determine mitigation strategy
            if request.mitigation_strategy == "auto":
                if di != 'inf' and di < 0.5:
                    strategy = "reweighing"
                elif di != 'inf' and di < 0.7:
                    strategy = "fairness_constraints"
                else:
                    strategy = "threshold_optimization"
            else:
                strategy = request.mitigation_strategy
            mlflow.log_param("selected_strategy", strategy)
            logging.info(f"Applying {strategy} mitigation strategy")
            
            # Apply mitigation
            if strategy == "threshold_optimization":
                # Quick mitigation through threshold adjustment
                result = await optimize_threshold(model_data,db,run.info.run_id)
            else:
                # Retrain with fairness constraints
                result = await retrain_with_fairness(
                    model_data, 
                    strategy, 
                    request.max_accuracy_loss,
                    db,
                    run.info.run_id
                )
            new_metrics = result["new_metrics"]
            if isinstance(new_metrics.get("disparate_impact"), float):
                if new_metrics["disparate_impact"] == float('inf'):
                    new_metrics["disparate_impact"] = "inf"
                elif new_metrics["disparate_impact"] == float('-inf'):
                    new_metrics["disparate_impact"] = "-inf"
                    
                    
                    
            new_accuracy = result["new_metrics"]["accuracy"]
            new_di = result["new_metrics"]["disparate_impact"]
            new_sp_diff = result["new_metrics"]["statistical_parity_diff"]
            
            mlflow.log_metric("new_accuracy", float(new_accuracy))
            
            #added a new block for post mitigation validation
            logging.info("Running post-mitigation validation...")
            
            validator = MitigationValidator(
                min_di_threshold=0.8,
                max_di_threshold=1.25,
                max_accuracy_loss=0.10,
                max_new_bias_increase=1.5
            )
            
            validation_result = validator.validate(
                original_metrics={
                    "accuracy": original_accuracy,
                    "disparate_impact": di,
                    "statistical_parity_diff": original_bias['Statistical_Parity'].get('statistical_parity_diff', 0),
                    "equal_opportunity_diff": original_bias['Equal_Opportunity'].get('difference', 0),
                    "average_odds_diff": original_bias['Average_Odds_Difference'].get('average_odds_difference', 0)
                },
                new_metrics={
                    "accuracy": new_accuracy,
                    "disparate_impact": new_di,
                    "statistical_parity_diff": new_sp_diff,
                    "equal_opportunity_diff": result["new_metrics"].get("equal_opportunity_diff", 0),
                    "average_odds_diff": result["new_metrics"].get("average_odds_diff", 0)
                },
                strategy=strategy
            )
            
            logging.info(
                f"Mitigation validation: score={validation_result['score']}, "
                f"success={validation_result['success']}"
            )
            
            # Log validation to MLflow
            mlflow.log_metric("mitigation_validation_score", validation_result["score"])
            mlflow.set_tag("mitigation_success", str(validation_result["success"]))
            mlflow.set_tag("mitigation_compliant", str(validation_result["detailed_checks"]["compliance"]["compliant"]))
            
            # Log detailed checks
            for check_name, check_result in validation_result["detailed_checks"].items():
                if "score" in check_result:
                    mlflow.log_metric(f"validation_{check_name}_score", check_result["score"])
            
            # Log issues and recommendations
            if validation_result["issues"]:
                mlflow.log_text(
                    "\n".join(validation_result["issues"]),
                    "validation_issues.txt"
                )
            
            if validation_result["recommendations"]:
                mlflow.log_text(
                    "\n".join(validation_result["recommendations"]),
                    "validation_recommendations.txt"
                )
            
            #added to handle better cases
            if isinstance(new_di, (int, float)) and new_di not in [float('inf'), float('-inf')]:
                mlflow.log_metric("new_disparate_impact", float(new_di))
            else:
                mlflow.log_metric("new_disparate_impact", 999.0)

            mlflow.log_metric("new_statistical_parity_diff", float(new_sp_diff))
            
            
            accuracy_change = result["accuracy_impact"]
            
            if isinstance(di, (int, float)) and isinstance(new_di, (int, float)):
                if di not in [float('inf'), float('-inf')] and new_di not in [float('inf'), float('-inf')]:
                    di_improvement = new_di - di
                else:
                    di_improvement = 0
            else:
                di_improvement = 0

            mlflow.log_metric("accuracy_change", float(accuracy_change))
            mlflow.log_metric("disparate_impact_improvement", float(di_improvement))
            
            
            di_improvement_pct = 0
            if isinstance(di, (int, float)) and di not in [float('inf'), float('-inf'), 0]:
                if isinstance(new_di, (int, float)) and new_di not in [float('inf'), float('-inf')]:
                    di_improvement_pct = ((new_di - di) / abs(di)) * 100
                    mlflow.log_metric("disparate_impact_improvement_pct", float(di_improvement_pct))
            
            accuracy_change_pct = 0
            if original_accuracy != 0:
                accuracy_change_pct = ((new_accuracy - original_accuracy) / original_accuracy) * 100
                mlflow.log_metric("accuracy_change_pct", float(accuracy_change_pct))
            
            # Calculate mitigation effectiveness score
            if accuracy_change != 0:
                effectiveness = abs(di_improvement / accuracy_change)
            else:
                effectiveness = abs(di_improvement * 100) if di_improvement != 0 else 0
            mlflow.log_metric("mitigation_effectiveness", float(effectiveness))
            
            # Tag success/failure
            if di_improvement > 0:
                mlflow.set_tag("mitigation_result", "improved")
                success_level = "high" if di_improvement > 0.1 else "moderate" if di_improvement > 0.05 else "low"
                mlflow.set_tag("improvement_level", success_level)
            else:
                mlflow.set_tag("mitigation_result", "no_improvement")
                
            di_str = f"{float(di):.3f}" if isinstance(di, (int, float)) and di not in [float('inf'), float('-inf')] else str(di)
            new_di_str = f"{float(new_di):.3f}" if isinstance(new_di, (int, float)) and new_di not in [float('inf'), float('-inf')] else str(new_di)           
            # Log mitigation summary
            summary = f"""
            Mitigation Summary:
            - Strategy: {strategy}
            - Original DI: {di_str}
            - New DI: {new_di_str}
            - DI Improvement: {di_improvement:.3f}
            - Original Accuracy: {original_accuracy:.3f}
            - New Accuracy: {new_accuracy:.3f}
            - Accuracy Change: {accuracy_change:.3f}
            - Effectiveness Score: {effectiveness:.2f}
            """
            mlflow.log_text(summary.strip(), "mitigation_summary.txt")
                        
            # Save mitigation report as artifact
            mitigation_report = {
                            "original_model_id": request.model_id,
                            "new_model_id": result.get("new_model_id", ""),
                            "strategy": strategy,
                            "original_metrics": {
                                "accuracy": float(original_accuracy),
                                "disparate_impact": float(di) if isinstance(di, (int, float)) and di not in [float('inf'), float('-inf')] else "inf",
                                "statistical_parity_diff": float(original_bias['Statistical_Parity'].get('statistical_parity_diff', 0))
                            },
                            "new_metrics": result["new_metrics"],
                            "improvements": {
                                "di_absolute": float(di_improvement),
                                "di_percentage": float(di_improvement_pct) if 'di_improvement_pct' in locals() else 0,
                                "accuracy_absolute": float(accuracy_change),
                                "accuracy_percentage": float(accuracy_change_pct) if 'accuracy_change_pct' in locals() else 0
                            },
                            "effectiveness": float(effectiveness),
                            "timestamp": datetime.now().isoformat()
                        }
            
            report_path = f"/tmp/mitigation_report_{request.model_id}.json"
            with open(report_path, 'w') as f:
                json.dump(mitigation_report, f, indent=2, default=str)
            mlflow.log_artifact(report_path)
            time.sleep(0.2)
            
            response_message = f"View in MLflow: http://localhost:5000/#/experiments/1/runs/{run.info.run_id}"
            
            # Add validation feedback to message
            if not validation_result["success"]:
                response_message = (
                    f" MITIGATION ISSUES DETECTED (Score: {validation_result['score']}/100)\n"
                    f"Issues: {'; '.join(validation_result['issues'][:2])}\n"
                    f"View details in MLflow: http://localhost:5000/#/experiments/1/runs/{run.info.run_id}"
                )
            elif validation_result["warnings"]:
                response_message = (
                    f"Mitigation successful (Score: {validation_result['score']}/100)\n"
                    f"Warnings: {'; '.join(validation_result['warnings'][:1])}\n"
                    f"View details in MLflow: http://localhost:5000/#/experiments/1/runs/{run.info.run_id}"
                )
            else:
                response_message = (
                    f"Mitigation fully successful (Score: {validation_result['score']}/100)\n"
                    f"Model ready for deployment. "
                    f"View in MLflow: http://localhost:5000/#/experiments/1/runs/{run.info.run_id}"
                )
            # Calculate improvements for response
            bias_improvement = {
                "disparate_impact": di_improvement,
                "statistical_parity_diff": abs(original_bias['Statistical_Parity'].get('statistical_parity_diff', 0)) - 
                                         abs(new_sp_diff)
            }
            
            logging.info(f"Mitigation MLflow run completed: {run.info.run_id}")
            
            mitigation_data = schemas.MitigationCreate(
                mitigation_id=f"mitigation_{uuid.uuid4().hex[:8]}",
                original_model_id=request.model_id,
                new_model_id=result["new_model_id"],
                strategy=strategy,
                original_accuracy=original_accuracy,
                new_accuracy=new_accuracy,
                original_disparate_impact=float(di) if isinstance(di, (int, float)) and di not in [float('inf'), float('-inf')] else None,
                new_disparate_impact=float(new_di) if isinstance(new_di, (int, float)) and new_di not in [float('inf'), float('-inf')] else None,
                accuracy_impact=result["accuracy_impact"],
                bias_improvement=bias_improvement,
                mlflow_run_id=run.info.run_id
            )
            
            crud.create_mitigation(db, mitigation_data)
            logging.info(f"Saved Mitigation Analysis model {request.model_id}")
            
            return MitigationResponse(
                status="success",
                original_model_id=request.model_id,
                new_model_id=result["new_model_id"],
                mlflow_run_id=run.info.run_id,
                strategy_applied=strategy,
                original_metrics={
                    "accuracy": original_accuracy,
                    "disparate_impact": safe_float(di),
                    "statistical_parity_diff": original_bias['Statistical_Parity'].get('statistical_parity_diff', 0)
                },
                new_metrics={"accuracy": new_accuracy,
                "disparate_impact": safe_float(new_di),
                "statistical_parity_diff":new_sp_diff,
                },
                bias_improvement=bias_improvement,
                accuracy_impact=result["accuracy_impact"],
                message=f"View in MLflow: http://localhost:5000/#/experiments/1/runs/{run.info.run_id}",
                
                validation_score=validation_result["score"],
                validation_success=validation_result["success"],
                validation_issues=validation_result["issues"],
                validation_warnings=validation_result["warnings"],
                validation_recommendations=validation_result["recommendations"],
            )
        
    except Exception as e:
        logging.error(f"Mitigation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
async def optimize_threshold(model_data: Dict,db: Session ,mlflow_run_id:str) -> Dict:
    """
    Optimize decision threshold for fairness without retraining
    """
    y_test = model_data["y_test"]
    y_prob = model_data.get("y_prob")
    s_test = model_data["s_test"]
    s_test_1d = s_test[:, 0] if s_test.ndim > 1 else s_test
    
    if y_prob is None:
        raise ValueError("Probability scores not available. Please retrain with probability output.")
    
    # For multi-class, convert to binary probabilities
    if len(y_prob.shape) > 1 and y_prob.shape[1] > 2:
        # Use max probability for threshold optimization
        y_prob = np.max(y_prob, axis=1)
    elif len(y_prob.shape) > 1:
        y_prob = y_prob[:, 1]
    
    best_threshold = 0.5
    best_di = 0
    best_metrics = None
    
    # Try different thresholds
    for threshold in np.arange(0.3, 0.8, 0.05):
        y_pred_new = (y_prob >= threshold).astype(int)
        metrics = generate_report_fairness(y_test, y_pred_new, s_test_1d)
        di = metrics['Disparate_Impact_Ratio'].get('ratio', 0)
        
        # Optimize for disparate impact closest to 1
        if di != 'inf' and abs(di - 1.0) < abs(best_di - 1.0):
            best_di = di
            best_threshold = threshold
            best_metrics = metrics
    
    # Apply best threshold
    y_pred_fair = (y_prob >= best_threshold).astype(int)
    new_accuracy = accuracy_score(y_test, y_pred_fair)
    original_accuracy = accuracy_score(y_test, model_data["y_pred"])
    
    # Create new model entry
    new_model_id = f"model_{uuid.uuid4().hex[:8]}"
    new_model_data = model_data.copy()
    new_model_data["y_pred"] = y_pred_fair
    new_model_data["optimal_threshold"] = best_threshold
    new_model_data["mitigation_applied"] = "threshold_optimization"
    
    store_model_data(new_model_id, new_model_data)
    
    try:
        threshold_optimized_model_db = schemas.ModelCreate(
            model_id=new_model_id,
            model_type=model_data["model_type"],
            task_type=model_data["task_type"],
            dataset_name=model_data.get("file_id", ""),
            target_column=model_data["target_column"],
            sensitive_columns=model_data["sensitive_columns"],
            feature_count=model_data.get("X_test").shape[1],
            training_samples=len(model_data.get("X_train", [])) if model_data.get("X_train") is not None else 0,
            test_samples=len(model_data.get("X_test", [])),
            accuracy=float(new_accuracy),
            mlflow_run_id=mlflow_run_id  
        )
        
        crud.create_model(db, threshold_optimized_model_db)
        logging.info(f"Saved threshold-optimized model {new_model_id} to database")
        
    except Exception as e:
        logging.error(f"Failed to save threshold-optimized model to database: {e}")
    
    return {
        "new_model_id": new_model_id,
        "new_metrics": {
            "accuracy": new_accuracy,
            "disparate_impact": safe_float(di),
            "statistical_parity_diff": best_metrics['Statistical_Parity'].get('statistical_parity_diff', 0)
        },
        "accuracy_impact": new_accuracy - original_accuracy
    }

async def retrain_with_fairness(model_data: Dict, strategy: str, max_accuracy_loss: float, db: Session,mlflow_run_id:str) -> Dict:
    """
    Retrain model with fairness constraints
    """
    start_time = time.time()
    
    # Get training data
    X_train = model_data.get("X_train")
    y_train = model_data.get("y_train")
    s_train = model_data.get("s_train")
    X_test = model_data["X_test"]
    y_test = model_data["y_test"]
    s_test = model_data["s_test"]
    
    if X_train is None or y_train is None:
        # Reload data from original file
        file_id = model_data.get("file_id") or model_data.get("dataset_id")
        if not file_id:
            raise ValueError("Cannot retrain - training data not available")
        
        logging.info(f"Reloading data from {file_id}")
        df = await get_file_from_s3(file_id)
        
        # Reprocess data
        X_train, X_test, y_train, y_test, s_train, s_test = preprocess_data(
            df,
            model_data["target_column"],
            model_data["sensitive_columns"],
            test_size=0.2
        )
    
    # Convert sensitive attributes to 1D for fairness weights
    s_train_1d = s_train[:, 0] if s_train.ndim > 1 else s_train
    s_test_1d = s_test[:, 0] if s_test.ndim > 1 else s_test
    
    if s_train_1d.dtype == object or s_train_1d.dtype.kind in ('U', 'S'):
        logging.info(f"Encoding categorical sensitive attribute: {s_train_1d.dtype}")
        le = LabelEncoder()
        s_train_1d = le.fit_transform(s_train_1d)
        s_test_1d = le.transform(s_test_1d)
    
    # Calculate fairness weights
    if strategy == "reweighing":
        weights = calculate_fairness_weights(y_train, s_train_1d, strategy="reweighing")
    elif strategy == "fairness_constraints":
        weights = calculate_fairness_weights(y_train, s_train_1d, strategy="demographic_parity")
    else:
        weights = np.ones(len(y_train))
    
    # Retrain model with same type
    model_type = model_data["model_type"]
    task_type = model_data["task_type"]
    
    logging.info(f"Retraining {model_type} with {strategy} strategy")
    
    if task_type in ["binary", "multiclass"]:
        # Import and train appropriate model
        if "XGBoost" in model_type:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif "Random Forest" in model_type:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif "Logistic" in model_type:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        else:
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            )
        
        # Train with fairness weights
        model.fit(X_train, y_train, sample_weight=weights)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
    else:
        raise ValueError("Fairness mitigation only supported for classification tasks")
    
    # Calculate new metrics
    new_bias = generate_report_fairness(y_test, y_pred, s_test_1d)
    new_accuracy = accuracy_score(y_test, y_pred)
    original_accuracy = accuracy_score(y_test, model_data["y_pred"])
    
    # Check if accuracy loss is acceptable
    accuracy_loss = original_accuracy - new_accuracy
    if accuracy_loss > max_accuracy_loss:
        logging.warning(f"Accuracy loss {accuracy_loss:.2%} exceeds maximum {max_accuracy_loss:.2%}")
    
    # Store new model
    new_model_id = f"model_{uuid.uuid4().hex[:8]}"
    new_model_data = {
        **model_data,
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob[:, 1] if y_prob is not None and y_prob.shape[1] == 2 else y_prob,
        "mitigation_applied": strategy,
        "fairness_weights_used": True,
        "retraining_time": time.time() - start_time
    }
    
    store_model_data(new_model_id, new_model_data)
    
    db = SessionLocal()
    try:
        mitigated_model_db = schemas.ModelCreate(
            model_id=new_model_id,
            model_type=model_data["model_type"],
            task_type=model_data["task_type"],
            dataset_name=model_data["file_id"],
            target_column=model_data["target_column"],
            sensitive_columns=model_data["sensitive_columns"],
            feature_count=model_data.get("X_test").shape[1],
            training_samples=len(model_data.get("X_train", [])),
            test_samples=len(model_data.get("X_test", [])),
            accuracy=float(new_accuracy),
            mlflow_run_id=mlflow_run_id 
        )
        
        crud.create_model(db, mitigated_model_db)
        logging.info(f"Saved mitigated model {new_model_id} to database")
    finally:
        db.close()
        
    return {
        "new_model_id": new_model_id,
        "new_metrics": {
            "accuracy": new_accuracy,
            "disparate_impact": clean_for_json(new_bias['Disparate_Impact_Ratio'].get('ratio', 1)),
            "statistical_parity_diff": new_bias['Statistical_Parity'].get('statistical_parity_diff', 0)
        },
        "accuracy_impact": new_accuracy - original_accuracy
    }

def assess_compliance(fairness_results: Dict) -> str:
    """
    Assess regulatory compliance based on fairness metrics
    """
    non_compliant = False
    warning = False
    
    for col_name, metrics in fairness_results.items():
        # Check disparate impact (80% rule)
        di = metrics['disparate_impact'].get('ratio', 1)
        if di != 'inf' and (di < 0.8 or di > 1.25):
            non_compliant = True
        elif di != 'inf' and (di < 0.85 or di > 1.18):
            warning = True
        
        # Check statistical parity
        sp_diff = metrics['statistical_parity'].get('statistical_parity_diff', 0)
        if abs(sp_diff) > 0.2:
            non_compliant = True
        elif abs(sp_diff) > 0.15:
            warning = True
    
    if non_compliant:
        return "NON_COMPLIANT - High bias detected"
    elif warning:
        return "WARNING - Moderate bias detected"
    else:
        return "COMPLIANT - Acceptable fairness levels"

def generate_recommendations(fairness_results: Dict, fairness_already_applied: bool = False) -> List[str]:
    """
    Generate specific recommendations based on detected bias
    """
    recommendations = []
    
    # Check if fairness was already applied
    if fairness_already_applied:
        recommendations.append("Note: This model was already trained with fairness constraints")
    
    for col_name, metrics in fairness_results.items():
        di_ratio = metrics['disparate_impact'].get('ratio', 1)
        
        # Based on disparate impact severity
        if di_ratio != 'inf':
            if di_ratio < 0.5:
                recommendations.append(f"CRITICAL: Apply reweighing mitigation for '{col_name}' - severe bias (DI={di_ratio:.2f})")
            elif di_ratio < 0.7:
                recommendations.append(f"HIGH: Use fairness constraints for '{col_name}' - significant bias (DI={di_ratio:.2f})")
            elif di_ratio < 0.8:
                recommendations.append(f"MODERATE: Try threshold optimization for '{col_name}' - mild bias (DI={di_ratio:.2f})")
        
        # Based on statistical parity
        sp_diff = metrics['statistical_parity'].get('statistical_parity_diff', 0)
        if abs(sp_diff) > 0.2:
            recommendations.append(f"Review features correlated with '{col_name}' - possible proxy discrimination")
        
        # Based on equal opportunity
        eo_diff = metrics['equal_opportunity'].get('difference', 0)
        if abs(eo_diff) > 0.15:
            recommendations.append(f"Consider separate thresholds for '{col_name}' groups to equalize TPR")
    
    if not recommendations:
        recommendations.append("Model shows acceptable fairness across all metrics")
    
    # Add actionable next steps
    if not fairness_already_applied and any("CRITICAL" in r or "HIGH" in r for r in recommendations):
        recommendations.append("ACTION: Use /bias/mitigate endpoint to retrain with fairness constraints")
    
    return recommendations[:5]  # Return top 5 recommendations



#