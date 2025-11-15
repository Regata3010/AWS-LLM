import pandas as pd
import numpy as np
import uuid
import json
import io
from datetime import datetime
from typing import Dict, Any, Tuple, List
import sys
import os

from backend.services.dataset_service import DatasetService
from core.src.exception import CustomException
from core.src.ingestion import upload_file_to_s3
from core.bias_detector.detector import generate_report_fairness
from core.bias_detector.preprocessor import preprocess_data
# from core.train_models.train_classifier import train_classification_model
# from core.train_models.train_regressor import train_regression_model
from core.bias_detector.aif360_wrapper import audit_with_aif360
from core.bias_detector.mitigation import apply_reweighing, apply_reject_option
from core.validation.classproblem import type_of_target

class BiasAnalysisService:
    """
    Complete bias analysis pipeline service.
    Orchestrates the entire workflow from dataset to comprehensive bias report.
    """
    
    def __init__(self):
        self.dataset_service = DatasetService()
        self.s3_bucket = "qwezxcbucket"
        self.reports_prefix = "bias_reports/"
        
        # Supported algorithms by task type
        self.classification_algorithms = [
            "Logistic Regression", "Random Forest", "XGBoost", "SVM", "Decision Tree"
        ]
        self.regression_algorithms = [
            "Linear Regression", "Random Forest Regressor", "XGBoost Regressor", "Ridge Regression", "Lasso Regression"
        ]
    
    async def complete_bias_analysis(self, dataset_id: str, model_algorithm: str = "auto") -> Dict[str, Any]:
        """
        MAIN FUNCTION: Complete end-to-end bias analysis pipeline.
        
        Workflow:
        1. Load dataset from S3
        2. Auto-detect problem type  
        3. Train appropriate model
        4. Run comprehensive fairness analysis
        5. AIF360 audit
        6. Generate complete report
        """
        try:
            analysis_id = str(uuid.uuid4())
            
            # Step 1: Load dataset and validate readiness
            df, metadata = await self.dataset_service.load_dataset_for_analysis(dataset_id)
            
            column_detection = metadata.get('column_detection', {})
            target_col = column_detection.get('target_column')
            sensitive_cols = column_detection.get('sensitive_columns', [])
            
            if not target_col or not sensitive_cols:
                raise ValueError("No valid target or sensitive columns detected")
            
            # Step 2: Preprocess data first
            X_train, X_test, y_train, y_test, s_train, s_test = preprocess_data(
                df, target_col, sensitive_cols, test_size=0.2, random_state=42
            )
            
            # Step 3: Auto-detect problem type using preprocessed target
            problem_type_analysis = self._detect_problem_type(y_train, target_col)
            task_type = problem_type_analysis['task_type']
            
            # Step 4: Select optimal algorithm based on problem type
            if model_algorithm == "auto":
                selected_algorithm = self._select_optimal_algorithm(task_type)
            else:
                selected_algorithm = model_algorithm
                # Validate algorithm matches task type
                if not self._validate_algorithm_for_task(selected_algorithm, task_type):
                    raise ValueError(f"Algorithm '{selected_algorithm}' not suitable for {task_type} task")
            
            # Step 5: Train model (classification or regression)
            if task_type in ["binary", "multiclass"]:
                model, y_pred = train_classification_model(
                    X_train, X_test, y_train, y_test, selected_algorithm
                )
                model_type = "classification"
            elif task_type == "continuous":
                model, y_pred = train_regression_model(
                    X_train, X_test, y_train, y_test, selected_algorithm
                )
                model_type = "regression"
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            # Step 6: Calculate model performance
            model_performance = self._calculate_model_performance(y_test, y_pred, model_type)
            
            # Step 7: Run comprehensive fairness analysis
            primary_sensitive = s_test if len(sensitive_cols) == 1 else s_test[:, 0]
            fairness_report = generate_report_fairness(
                y_test, y_pred, primary_sensitive, 
                model_name=f"{selected_algorithm}_{dataset_id[:8]}"
            )
            
            # Step 8: AIF360 audit (only for binary classification)
            aif360_audit = {}
            if task_type == "binary":
                try:
                    aif360_audit = audit_with_aif360(df, target_col, sensitive_cols[0])
                except Exception as e:
                    aif360_audit = {"error": f"AIF360 audit failed: {str(e)}"}
            else:
                aif360_audit = {"message": f"AIF360 audit not available for {task_type} tasks"}
            
            # Step 9: Generate comprehensive report
            complete_report = {
                'analysis_id': analysis_id,
                'dataset_info': {
                    'dataset_id': dataset_id,
                    'filename': metadata.get('filename'),
                    'shape': metadata.get('shape'),
                    'data_quality_score': metadata.get('data_quality', {}).get('quality_score', 0)
                },
                'problem_analysis': problem_type_analysis,
                'column_selection': {
                    'target_column': target_col,
                    'sensitive_columns': sensitive_cols,
                    'llm_reasoning': column_detection.get('llm_reasoning', {})
                },
                'model_training': {
                    'algorithm': selected_algorithm,
                    'model_type': model_type,
                    'task_type': task_type,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test)
                },
                'model_performance': model_performance,
                'fairness_analysis': fairness_report,
                'aif360_audit': aif360_audit,
                'overall_assessment': self._generate_overall_assessment(fairness_report, aif360_audit, model_performance),
                'analysis_timestamp': datetime.now().isoformat(),
                'recommendations': self._generate_comprehensive_recommendations(fairness_report, aif360_audit, problem_type_analysis)
            }
            
            # Step 10: Store complete report in S3
            await self._store_analysis_report(analysis_id, complete_report)
            
            return complete_report
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _detect_problem_type(self, y_train: np.ndarray, target_col: str) -> Dict[str, Any]:
        """
        Auto-detect if this is classification or regression task.
        """
        try:
            task_type = type_of_target(y_train)
            unique_values = len(np.unique(y_train))
            
            analysis = {
                'task_type': task_type,
                'unique_target_values': unique_values,
                'target_column': target_col,
                'suitable_for_bias_analysis': task_type in ["binary", "multiclass"],
                'recommended_approach': self._get_recommended_approach(task_type, unique_values)
            }
            
            if task_type == "continuous":
                analysis['note'] = "Regression task - limited bias analysis available"
            elif task_type == "binary":
                analysis['note'] = "Binary classification - full bias analysis available"
            elif task_type == "multiclass":
                analysis['note'] = "Multi-class classification - adapted bias analysis available"
            
            return analysis
            
        except Exception as e:
            return {
                'task_type': 'unknown',
                'error': str(e),
                'suitable_for_bias_analysis': False
            }
    
    def _select_optimal_algorithm(self, task_type: str) -> str:
        """Select best algorithm for the detected task type."""
        optimal_algorithms = {
            'binary': 'Logistic Regression',
            'multiclass': 'Random Forest', 
            'continuous': 'Random Forest Regressor'
        }
        return optimal_algorithms.get(task_type, 'Logistic Regression')
    
    def _validate_algorithm_for_task(self, algorithm: str, task_type: str) -> bool:
        """Validate if algorithm is suitable for task type."""
        if task_type in ["binary", "multiclass"]:
            return algorithm in self.classification_algorithms
        elif task_type == "continuous":
            return algorithm in self.regression_algorithms
        return False
    
    def _calculate_model_performance(self, y_test: np.ndarray, y_pred: np.ndarray, model_type: str) -> Dict[str, float]:
        """Calculate comprehensive model performance metrics."""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
            
            if model_type == "classification":
                return {
                    'accuracy': round(accuracy_score(y_test, y_pred), 4),
                    'precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
                    'recall': round(recall_score(y_test, y_pred, average='weighted'), 4),
                    'f1_score': round(f1_score(y_test, y_pred, average='weighted'), 4)
                }
            else:  # regression
                return {
                    'mse': round(mean_squared_error(y_test, y_pred), 4),
                    'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                    'r2_score': round(r2_score(y_test, y_pred), 4),
                    'mae': round(np.mean(np.abs(y_test - y_pred)), 4)
                }
                
        except Exception as e:
            return {'error': f"Performance calculation failed: {str(e)}"}
    
    def _generate_overall_assessment(self, fairness_report: Dict, aif360_audit: Dict, model_performance: Dict) -> Dict[str, Any]:
        """Generate executive summary assessment."""
        try:
            # Extract bias severity from fairness report
            bias_severities = []
            if isinstance(fairness_report, dict):
                for metric_name, metric_data in fairness_report.items():
                    if isinstance(metric_data, dict) and 'severity' in metric_data:
                        bias_severities.append(metric_data['severity'])
            
            # Overall bias assessment
            if 'HIGH' in bias_severities:
                bias_status = "HIGH_RISK"
            elif bias_severities.count('MODERATE') >= 2:
                bias_status = "MODERATE_RISK"
            else:
                bias_status = "LOW_RISK"
            
            # Model performance assessment
            performance_status = "GOOD"
            if 'accuracy' in model_performance:
                if model_performance['accuracy'] < 0.7:
                    performance_status = "POOR"
                elif model_performance['accuracy'] < 0.8:
                    performance_status = "MODERATE"
            elif 'r2_score' in model_performance:
                if model_performance['r2_score'] < 0.5:
                    performance_status = "POOR"
                elif model_performance['r2_score'] < 0.7:
                    performance_status = "MODERATE"
            
            return {
                'bias_risk_level': bias_status,
                'model_performance_level': performance_status,
                'recommendation': self._get_overall_recommendation(bias_status, performance_status),
                'compliance_status': 'REVIEW_REQUIRED' if bias_status == 'HIGH_RISK' else 'ACCEPTABLE',
                'metrics_analyzed': len([s for s in bias_severities if s]),
                'high_risk_metrics': bias_severities.count('HIGH'),
                'aif360_available': 'error' not in aif360_audit and 'message' not in aif360_audit
            }
            
        except Exception as e:
            return {
                'error': f"Assessment generation failed: {str(e)}",
                'bias_risk_level': 'UNKNOWN',
                'model_performance_level': 'UNKNOWN'
            }
    
    def _get_overall_recommendation(self, bias_status: str, performance_status: str) -> str:
        """Generate overall recommendation based on bias and performance."""
        if bias_status == "HIGH_RISK":
            return "IMMEDIATE ACTION REQUIRED: Implement bias mitigation techniques before deployment"
        elif bias_status == "MODERATE_RISK" and performance_status == "GOOD":
            return "MONITOR CLOSELY: Consider bias mitigation while maintaining model performance"
        elif bias_status == "LOW_RISK" and performance_status == "GOOD":
            return "APPROVED FOR DEPLOYMENT: Continue monitoring bias metrics over time"
        elif performance_status == "POOR":
            return "IMPROVE MODEL: Address both performance and bias issues before deployment"
        else:
            return "REVIEW AND OPTIMIZE: Balance performance and fairness objectives"
    
    def _get_recommended_approach(self, task_type: str, unique_values: int) -> str:
        """Get recommended approach for the detected problem type."""
        if task_type == "binary":
            return "Binary classification - full bias analysis with all fairness metrics"
        elif task_type == "multiclass":
            return "Multi-class classification - adapted bias analysis (one-vs-rest approach)"
        elif task_type == "continuous":
            return "Regression task - limited to correlation-based bias metrics"
        else:
            return f"Unknown task type with {unique_values} unique values - manual review required"
    
    def _generate_comprehensive_recommendations(self, fairness_report: Dict, aif360_audit: Dict, problem_analysis: Dict) -> Dict[str, List[str]]:
        """Generate detailed recommendations by category."""
        recommendations = {
            'immediate_actions': [],
            'monitoring_requirements': [],
            'mitigation_strategies': [],
            'compliance_considerations': []
        }
        
        try:
            # Extract bias issues from fairness report
            high_bias_metrics = []
            moderate_bias_metrics = []
            
            if isinstance(fairness_report, dict):
                for metric_name, metric_data in fairness_report.items():
                    if isinstance(metric_data, dict) and 'severity' in metric_data:
                        if metric_data['severity'] == 'HIGH':
                            high_bias_metrics.append(metric_name)
                        elif metric_data['severity'] == 'MODERATE':
                            moderate_bias_metrics.append(metric_name)
            
            # Immediate actions for high bias
            if high_bias_metrics:
                recommendations['immediate_actions'].extend([
                    f"Address high bias in: {', '.join(high_bias_metrics)}",
                    "Consider data preprocessing techniques (reweighing, synthetic data)",
                    "Review training data for representation issues",
                    "Implement fairness constraints during model training"
                ])
            
            # Monitoring for moderate bias
            if moderate_bias_metrics:
                recommendations['monitoring_requirements'].extend([
                    f"Monitor moderate bias in: {', '.join(moderate_bias_metrics)}",
                    "Set up regular bias monitoring in production",
                    "Track bias metrics over time for degradation"
                ])
            
            # Mitigation strategies
            task_type = problem_analysis.get('task_type')
            if task_type == "binary":
                recommendations['mitigation_strategies'].extend([
                    "Pre-processing: Reweighing, synthetic data generation",
                    "In-processing: Fairness constraints, adversarial debiasing", 
                    "Post-processing: Threshold optimization, calibration"
                ])
            
            # Compliance considerations
            if high_bias_metrics or len(moderate_bias_metrics) >= 2:
                recommendations['compliance_considerations'].extend([
                    "Document bias analysis for compliance audits",
                    "Review against anti-discrimination regulations",
                    "Consider legal review before production deployment",
                    "Implement bias monitoring dashboard for ongoing compliance"
                ])
            
            return recommendations
            
        except Exception as e:
            return {
                'error': [f"Recommendation generation failed: {str(e)}"],
                'immediate_actions': ['Manual review required'],
                'monitoring_requirements': ['Set up basic monitoring'],
                'mitigation_strategies': ['Consult bias mitigation literature'],
                'compliance_considerations': ['Legal review recommended']
            }
    
    async def _store_analysis_report(self, analysis_id: str, report: Dict[str, Any]):
        """Store complete analysis report in S3."""
        try:
            report_s3_key = f"{self.reports_prefix}{analysis_id}_complete_analysis.json"
            
            report_json = json.dumps(report, indent=2, default=str)
            report_buffer = io.BytesIO(report_json.encode('utf-8'))
            
            upload_file_to_s3(file_obj=report_buffer, bucket=self.s3_bucket, s3_key=report_s3_key)
            
            # Update report with S3 location
            report['report_s3_location'] = report_s3_key
            
        except Exception as e:
            # Don't fail the whole analysis if storage fails
            report['storage_error'] = f"Failed to store report: {str(e)}"