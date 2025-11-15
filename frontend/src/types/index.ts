
export interface Model {
  // Common fields (both types)
  model_id: string;
  model_type: string;
  bias_status: 'compliant' | 'warning' | 'critical' | 'unknown';
  created_at: string;
  updated_at: string;
  source: 'external' | 'trained';
  platform: string;
  has_analysis: boolean;
  sensitive_columns: string[];
  
  // Trained model fields (optional)
  task_type?: string;
  dataset_name?: string;
  target_column?: string;
  feature_count?: number;
  training_samples?: number;
  test_samples?: number;
  accuracy?: number;
  mlflow_run_id?: string;
  is_mitigated?: boolean;
  parent_model_id?: string;
  mitigation_strategy?: string;
  
  // External model fields (optional)
  model_name?: string;
  description?: string;
  framework?: string;
  version?: string;
  sensitive_attributes?: string[];
  predictions_logged?: number;
  monitoring_enabled?: boolean;
  status?: string;
  endpoint_url?: string;
  alert_thresholds?: Record<string, number>;
}


export interface ModelsResponse {
  total: number;
  showing: number;
  skip: number;
  limit: number;
  models: Model[];
}

export interface BiasMetrics {
  statistical_parity: {
    metric: string;
    statistical_parity_diff: number;
    group_0_rate: number;
    group_1_rate: number;
    bias_detected: boolean;
    threshold: number;
    severity: string;
    interpretation: string;
  };
  disparate_impact: {
    metric: string;
    ratio: number | string;
    group_0_rate: number;
    group_1_rate: number;
    bias_detected: boolean;
    threshold: number[];
    severity: string;
    interpretation: string;
  };
  equal_opportunity: {
    metric: string;
    difference: number;
    group_0_tpr: number;
    group_1_tpr: number;
    bias_detected: boolean;
    threshold: number;
    severity: string;
    interpretation: string;
  };
  average_odds: {
    metric: string;
    average_difference: number;
    group_0_tpr: number;
    group_0_fpr: number;
    group_1_tpr: number;
    group_1_fpr: number;
    bias_detected: boolean;
    threshold: number;
    severity: string;
    interpretation: string;
  };
  fpr_parity: any;
  predictive_parity: any;
  treatment_equality: any;
}

export interface BiasAnalysisCreate {
  analysis_id: string;
  model_id: string;
  compliance_status: string;
  bias_status: string;
  fairness_metrics: Record<string, BiasMetrics>;
  aif360_metrics?: Record<string, any>;
  recommendations: string[];
  mlflow_run_id: string;
}

export interface BiasAnalysis extends BiasAnalysisCreate {
  analyzed_at: string;
}

export interface BiasHistoryResponse {
  model_id: string;
  total_analyses: number;
  history: BiasAnalysis[];
}

export interface MitigationCreate {
  mitigation_id: string;
  original_model_id: string;
  new_model_id: string;
  strategy: string;
  original_accuracy: number;
  new_accuracy: number;
  original_disparate_impact?: number;
  new_disparate_impact?: number;
  accuracy_impact: number;
  bias_improvement: Record<string, number>;
  mlflow_run_id: string;
}

export interface Mitigation extends MitigationCreate {
  created_at: string;
}

export interface DatasetUploadRequest {
  // FormData - no interface needed, file is uploaded directly
}

export interface ColumnAnalysisRequest {
  file_id: string;
  use_llm?: boolean;
  domain?: string;
}

export interface ColumnSelectionRequest {
  file_id: string;
  use_llm?: boolean;
  domain?: string;
}

export interface ColumnOverrideRequest {
  file_id: string;
  target_column: string;
  sensitive_columns: string[];
}

export interface TrainingRequest {
  file_id: string;
  target_column: string;
  sensitive_columns: string[];
  model_type?: string;
  test_size?: number;
  enable_fairness?: boolean;
  fairness_strategy?: string;
}

export interface BiasDetectionRequest {
  model_id: string;
}

export interface MitigationRequest {
  model_id: string;
  mitigation_strategy?: 'auto' | 'reweighing' | 'threshold_optimization' | 'fairness_constraints';
  max_accuracy_loss?: number;
}

export interface DetectTaskTypeRequest {
  file_id: string;
  target_column: string;
}

export interface DatasetUploadResponse {
  file_id: string;
  filename: string;
  rows: number;
  columns: number;
  s3_path: string;
  upload_timestamp: string;
  file_size_mb: number;
//   column_names : string[];
}

export interface ColumnAnalysisResponse {
  file_id: string;
  suggested_target?: string;
  suggested_sensitive: string[];
  all_columns: string[];
  column_metadata: Record<string, Record<string, any>>;
  llm_analysis?: Record<string, any>;
  manual_mode: boolean;
}

export interface ColumnOverrideResponse {
  file_id: string;
  target_column: string;
  sensitive_columns: string[];
  validation: Record<string, any>;
  ready_for_analysis: boolean;
}

export interface TrainingResponse {
  model_id: string;
  task_type: string;
  model_type: string;
  metrics: {
    accuracy: number;
    f1_score?: number;
    precision?: number;
    recall?: number;
    mse?: number;
    mae?: number;
    r2_score?: number;
    rmse?: number;
  };
  training_samples: number;
  testing_samples: number;
  message: string;
}

export interface BiasDetectionResponse {
  model_id: string;
  mlflow_run_id: string;
  fairness_metrics: Record<string, BiasMetrics>;
  aif360_metrics: {
    'Mean Difference': number;
    'Disparate Impact': number;
    'Consistency': number;
  } | null;
  compliance_status: string;
  recommendations: string[];
  task_type: string;
  fairness_applied: boolean;
  latency?: number;
}

export interface MitigationResponse {
  status: string;
  original_model_id: string;
  new_model_id: string;
  mlflow_run_id: string;
  strategy_applied: string;
  original_metrics: {
    accuracy: number;
    disparate_impact: number | string;
    statistical_parity_diff: number;
  };
  new_metrics: {
    accuracy: number;
    disparate_impact: number | string;
    statistical_parity_diff: number;
  };
  bias_improvement: Record<string, number>;
  accuracy_impact: number;
  message: string;
  // optional created timestamp returned by the mitigation endpoint
  created_at?: string | number | null;
}

export interface DetectTaskTypeResponse {
  task_type: string;
  recommended_models: string[];
  default_model: string;
  target_stats: Record<string, any>;
}

export interface DashboardSummary {
  total_models: number;
  compliant_models: number;
  models_at_risk: number;
  critical_models: number;
  unknown_status: number;
  total_analyses: number;
  compliance_rate: number;
}

export interface RecentActivityResponse {
  recent_models: Model[];
}

export interface ColumnSelectionResponse {
  file_id: string;
  all_columns: string[];
  column_metadata: Record<string, Record<string, any>>;
  
  // ✅ Performance tracking
  analysis_time_seconds?: number;
  tokens?: {
    input: number;
    output: number;
    total: number;
    cost_usd: number;
    model: string;
  };
  
  // ✅ LLM/Heuristic suggestions
  suggested_target?: string;
  suggested_sensitive?: string[];
  target_confidence?: number;
  target_rationale?: string;
  sensitive_details?: Array<{
    column: string;
    protected_class: string;
    attribute_type?: string;
    confidence: number;
    reasoning?: {
      why_sensitive: string;
      legal_basis: string;
      historical_discrimination?: string;
    };
    risk_assessment?: {
      discrimination_risk: string;
      correlation_with_target?: number;
      intersectional_concerns?: string[];
    };
    mitigation_priority?: number;
  }>;
  llm_warnings?: string[];
  
  // ✅ User manual selection (confirmed)
  selected_target?: string;
  selected_sensitive?: string[];
  
  // ✅ Proxy detection
  proxy_variables?: Array<{
    column: string;
    proxies_for: string[];
    evidence?: string;
    recommendation?: string;
    score?: number;
    method?: string;
    risk_level?: string;
  }>;
  
  proxy_detection_summary?: {
    total_proxies_detected: number;
    high_risk_count: number;
    high_risk_features: string[];
    medium_risk_count: number;
    medium_risk_features: string[];
    tests_performed: number;
    tests_skipped: number;
    recommendations: string[];
  };
  
  // ✅ Advanced analysis
  intersectionality_analysis?: {
    high_risk_combinations?: Array<{
      attributes: string[];
      amplification_factor?: string;
      evidence?: string;
    }>;
    recommended_subgroup_analysis?: string[];
  };
  
  compliance_notes?: {
    cfpb_requirements?: string;
    eeoc_requirements?: string;
    eu_ai_act?: string;
    state_laws?: string[];
  };
  
  recommended_metrics?: string[];
  
  full_llm_analysis?: {
    analysis_metadata?: Record<string, any>;
    target_variable?: Record<string, any>;
    sensitive_attributes?: Array<Record<string, any>>;
    proxy_variables_detected?: Array<Record<string, any>>;
    intersectionality_analysis?: Record<string, any>;
    data_quality_warnings?: string[];
    recommended_fairness_metrics?: string[];
    regulatory_compliance_notes?: Record<string, any>;
    column_usage_recommendations?: Record<string, any>;
    case_study_references?: string[];
  };
  
  // ✅ Validation results
  validation: {
    valid: boolean;
    errors: string[];
    warnings: string[];
    recommendations?: string[];
    can_proceed: boolean;
  };
  
  detection_method: string;
  ready_for_training: boolean;
  data_quality_issues?: string[] | null;
}

export interface MitigationInfoResponse {
  original_model_id: string;
  strategy: string;
  original_accuracy: number;
  new_accuracy: number;
  accuracy_impact: number;
  bias_improvement: Record<string, number>;
  created_at: string;
  mlflow_run_id: string;
}

export type BiasStatus = 'compliant' | 'warning' | 'critical' | 'unknown';
export type TaskType = 'binary' | 'multiclass' | 'continuous';
export type MitigationStrategy = 'auto' | 'reweighing' | 'threshold_optimization' | 'fairness_constraints';


//ws runs here 
export interface DriftStatus {
  drift_score: number;
  status: 'stable' | 'warning' | 'drifting';
  trend?: 'increasing' | 'stable' | 'decreasing';
  velocity?: number;
}

export interface Alert {
  type: 'critical' | 'warning' | 'info';
  metric: string;
  attribute?: string;
  value?: number;
  threshold?: number;
  message: string;
  regulation?: string;
}

export interface ModelMetrics {
  model_id: string;
  model_name?: string;
  predictions_last_hour?: number;
  avg_confidence?: number;
  drift_status?: DriftStatus;  // Now properly typed
  current_bias_metrics?: {
    disparate_impact: number;
    statistical_parity: number;
    equal_opportunity: number;
    accuracy: number;
  };
  alerts: Alert[];
  trends?: Record<string, TrendData>;      // NEW: Fairness metric trends
  performance?: PerformanceMetrics;
  timestamp: string;
}

export interface SystemHealth {
  api_latency_ms: number;
  active_connections: number;
  predictions_per_second: number;
}

export interface ModelStatus {
  model_id: string;
  model_type: string;
  status: 'healthy' | 'warning' | 'critical';
  drift_score: number;
  last_updated: string;
}

export interface DashboardData {
  timestamp: string;
  total_models: number;
  models_at_risk: number;
  system_health: SystemHealth;
  models_status: ModelStatus[];
  recent_alerts?: Alert[];
}

// export interface Model {
//   model_id: string;
//   model_type: string;
//   accuracy: number;
//   dataset_name: string;
//   bias_status: 'compliant' | 'warning' | 'critical' | 'unknown';
//   created_at: string;
// }

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';


// ab_testing runs here
export interface ModelAB{
  model_id: string;
  model_type: string;
  accuracy: number;
  dataset_name: string;
}

export interface ABTest {
  test_id: string;
  test_name: string;
  model_a: {
    model_id: string;
    model_type: string;
    baseline_accuracy: number;
  };
  model_b: {
    model_id: string;
    model_type: string;
    baseline_accuracy: number;
  };
  traffic_split: number;
  status: string;
  created_at: string;
  results: {
    model_a: {
      predictions: number;
      correct: number;
      accuracy: number;
      bias_violations: number;
    };
    model_b: {
      predictions: number;
      correct: number;
      accuracy: number;
      bias_violations: number;
    };
    statistical_significance?: {
      p_value: number;
      confidence: number;
      significant: boolean;
    };
    winner?: string;
  };
}

export interface TrendData {
  direction: 'improving' | 'degrading' | 'stable';
  change_rate: number;
  recent_avg: number;
  baseline_avg: number;
}

export interface PerformanceMetrics {
  baseline_accuracy: number;
  current_accuracy: number;
  accuracy_drop: number;
}

// Version 2.0 Endpoints for Frntend
export interface ExternalModel {
  model_id: string;
  model_name: string;
  description?: string;
  model_type: string;  // "classification" | "regression"
  framework?: string;  // "sklearn", "xgboost", "tensorflow", etc.
  version?: string;
  endpoint_url?: string;
  sensitive_attributes: string[];  // Changed from sensitive_columns to match backend
  monitoring_enabled: boolean;
  alert_thresholds?: Record<string, number>;
  status: 'active' | 'paused' | 'archived';
  organization_id?: string;
  created_by?: string;
  created_at: string;
  updated_at: string;
}

export interface RegisterExternalModelRegisterRequest {
  model_name: string;
  model_type: string;
  framework?: string;
  version?: string;
  description?: string;
  sensitive_attributes: string[];
  // monitoring_enabled?: boolean;
  endpoint_url?: string;
  alert_thresholds?: Record<string, number>;
}


export interface RegisterExternalModelRegisterResponse {
  model_id: string;
  model_name: string;
  status: string;
  message: string;
  next_steps?: string[];
}

export interface PredictionLogRequest {
  model_id: string;
  prediction: number;
  prediction_proba?: number;
  ground_truth?: number;
  sensitive_attributes: Record<string, any>;
  features?: Record<string, any>;
}

export interface PredictionLogResponse {
  status:string;
  log_id:string;
  logged_at:string;
}

export interface  BatchPredictionLogRequest {
  model_id:string;
  predictions : Record<string, any>[];
  predictions_proba?: Record<string, any>[];
  ground_truths?: Record<string, any>[];
  sensitive_attributes_list: Record<string, any>[];
  features_list?: Record<string, any>[];
}

export interface BatchPredictionLogResponse {
  status:string;
  batch_id : string;
  prediction_logged : number;
  model_id : string;
  logged_at : string;
}

export interface AnalyzeRequest {
  model_id:string;
  period_days?:number;
  min_samples?:number;
}

export interface AnalyzeResponse {
  analysis_id:string;
  mlflow_run_id:string;
  model_id:string;
  period?:number;
  fairness_metrics: Record<string, BiasMetrics>;
  intersectionality: Record<string, any>;
  compliance_status:string;
  recommendations:string[];
  analyzed_at:string;
  has_ground_truth:boolean;
  bias_status:string;
}

export interface ReportGenerateRequest{
  report_type : string;
  include_recommendations?: boolean;
  include_technical_details?: boolean;
  format?: string;
}

export interface ReportGenerateResponse{
  report_id : string;
  download_url : string;
  generated_at : string;
  model_id : string;
  report_type : string;
  file_size_mb?: number;
  llm_summary?: Record<string, any>;
}

export interface EnhancedDriftStatus {
  drift_score: number;
  status: 'stable' | 'warning' | 'critical';
  trend: 'increasing' | 'stable' | 'decreasing';
  velocity: number;
}

export interface TrendData {
  direction: 'improving' | 'degrading' | 'stable';
  change_rate: number;
  recent_avg: number;
  baseline_avg: number;
}

export interface PerformanceMetrics {
  baseline_accuracy: number;
  current_accuracy: number;
  accuracy_drop: number;
}

export interface EnhancedAlert {
  type: 'critical' | 'warning' | 'info';
  metric: string;
  attribute?: string;
  value: number;
  threshold: number | string;
  message: string;
  regulation?: string;
}

export interface EnhancedModelMetrics {
  timestamp: string;
  model_id: string;
  model_name: string;
  predictions_last_hour: number;
  avg_confidence: number;
  drift_status: EnhancedDriftStatus;
  current_bias_metrics: {
    disparate_impact: number;
    statistical_parity: number;
    equal_opportunity: number;
    accuracy: number;
  };
  trends: Record<string, TrendData>;
  performance: PerformanceMetrics;
  alerts: EnhancedAlert[];
}

//Ai Agents types
export interface ChatPanelProps {
  isOpen: boolean;
  onClose: () => void;
  modelId?: string;
}

export interface ChatWidgetProps {
  modelId?: string;
}

export interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export interface ToolUsageIndicatorProps {
  tools: string[];
}

// export interface ChatMessage {
//   id: string;
//   role: 'user' | 'assistant';
//   content: string;
//   timestamp: string;
//   tools_used?: string[];
// }

export interface ChatRequest {
  message: string;
  model_id?: string;
  thread_id?: string;
}

export interface ChatResponse {
  response: string;
  tools_used: string[];
  timestamp: string;
}