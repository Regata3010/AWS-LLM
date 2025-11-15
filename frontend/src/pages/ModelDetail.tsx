import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  ArrowLeft, 
  ExternalLink, 
  Trash2,
  Play,
  RefreshCw,
  AlertCircle,
  Sparkles,
  ArrowRight,
  CheckCircle,
  FileText,
  
} from 'lucide-react';
import { modelApi, biasApi } from '../services/api';
import Card from '../components/ui/Card';
import Badge, { getBiasStatusVariant } from '../components/ui/Badge';
import MetricDisplay from '../components/features/MetricDisplay';
import type { MitigationRequest, MitigationInfoResponse } from '../types';

const ModelDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  
  const [selectedStrategy, setSelectedStrategy] = useState<string>('auto');
  const [maxAccuracyLoss, ] = useState<number>(0.05);
  const [showMitigationSuccess, setShowMitigationSuccess] = useState(false);
  const [newModelId, setNewModelId] = useState<string>('');

  // Fetch model details
  const { data: model, isLoading: modelLoading } = useQuery({
    queryKey: ['model', id],
    queryFn: () => modelApi.getById(id!),
    enabled: !!id,
  });

  // Determine model type
  const isExternalModel = model?.source === 'external';
  const isTrainedModel = !isExternalModel;

  // Only fetch mitigation for trained models
  const { data: mitigationInfo } = useQuery<MitigationInfoResponse | null>({
    queryKey: ['mitigation-info', id],
    queryFn: () => modelApi.getMitigationInfo(id!),
    enabled: !!id && isTrainedModel,
    retry: false,
  });

  const mi = mitigationInfo as MitigationInfoResponse | null;

  // Fetch bias analysis
  const { 
    data: biasAnalysis, 
    isLoading: biasLoading,
    error: biasError 
  } = useQuery({
    queryKey: ['bias-latest', id],
    queryFn: () => biasApi.getLatest(id!),
    enabled: !!id,
    retry: false,
    throwOnError: (error: any) => error?.response?.status !== 404
  });

  const detectBiasMutation = useMutation({
    mutationFn: () => biasApi.detect({ model_id: id! }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['bias-latest', id] });
      queryClient.invalidateQueries({ queryKey: ['model', id] });
    },
  });

  const mitigateMutation = useMutation({
    mutationFn: (request: MitigationRequest) => biasApi.mitigate(request),
    onSuccess: (data) => {
      setNewModelId(data.new_model_id);
      setShowMitigationSuccess(true);
      setTimeout(() => navigate(`/model/${data.new_model_id}`), 1500);
    },
    onError: (error: any) => {
      alert(`Mitigation failed: ${error.response?.data?.detail || error.message}`);
    }
  });

  const deleteMutation = useMutation({
    mutationFn: () => modelApi.delete(id!),
    onSuccess: () => navigate('/'),
  });

  if (modelLoading || !model) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
          <p className="text-gray-400">Loading model details...</p>
        </div>
      </div>
    );
  }

  const handleMitigation = () => {
    if (!id) return;
    mitigateMutation.mutate({
      model_id: id,
      mitigation_strategy: selectedStrategy as any,
      max_accuracy_loss: maxAccuracyLoss,
    });
  };

  const hasBiasAnalysis = !!biasAnalysis && !biasError;
  const firstSensitiveCol = Object.keys(biasAnalysis?.fairness_metrics || {})[0];
  const metrics = biasAnalysis?.fairness_metrics?.[firstSensitiveCol];

  return (
    <div className="min-h-screen bg-background-primary">
      {/* Success Banner */}
      {showMitigationSuccess && (
        <div className="fixed top-4 right-4 z-50 max-w-md">
          <div className="bg-green-900/90 backdrop-blur border border-green-600 rounded-lg p-4 shadow-2xl">
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-green-400" />
              <div>
                <h4 className="text-white font-semibold mb-1">Mitigation Complete!</h4>
                <p className="text-sm text-green-200">New model: {newModelId.slice(0, 16)}...</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="px-8 py-6 border-b border-gray-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/')}
              className="p-2 hover:bg-background-card rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-gray-400" />
            </button>
            <div>
              <h1 className="text-3xl font-bold text-white">
                {model.model_name || model.model_id}
              </h1>
              <p className="text-gray-400 mt-1">
                {model.model_type} • {isExternalModel ? 'External Model' : 'Trained Model'}
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <Badge variant={getBiasStatusVariant(model.bias_status)} size="lg">
              {model.bias_status || 'Not Analyzed'}
            </Badge>
            <button
              onClick={() => {
                if (confirm('Are you sure you want to delete this model?')) {
                  deleteMutation.mutate();
                }
              }}
              className="px-4 py-2 bg-status-danger/20 text-status-danger rounded-lg hover:bg-status-danger/30 transition-colors flex items-center gap-2"
            >
              <Trash2 className="w-4 h-4" />
              Delete
            </button>
          </div>
        </div>
      </div>

      <div className="p-8">
        {/* Mitigation Info Banner (trained models only) */}
        {mi && isTrainedModel && (
          <div className="mb-6 p-4 bg-primary/10 border border-primary/30 rounded-lg">
            <div className="flex items-start gap-3">
              <Sparkles className="w-5 h-5 text-primary light" />
              <div className="flex-1">
                <h4 className="text-white font-semibold mb-3">
                  This model was created through bias mitigation
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-gray-400 mb-1">Strategy</p>
                    <p className="text-white font-medium capitalize">
                      {mi.strategy.replace(/_/g, ' ')}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-400 mb-1">Accuracy Impact</p>
                    <p className={`font-medium ${mi.accuracy_impact >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {mi.accuracy_impact >= 0 ? '+' : ''}{(mi.accuracy_impact * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-400 mb-1">Original Model</p>
                    <button
                      onClick={() => navigate(`/model/${mi.original_model_id}`)}
                      className="text-indigo-400 hover:text-indigo-300 font-medium flex items-center gap-1"
                    >
                      View <ArrowRight className="w-3 h-3" />
                    </button>
                  </div>
                  <div>
                    <p className="text-gray-400 mb-1">Created</p>
                    <p className="text-white">{new Date(mi.created_at).toLocaleDateString()}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Model Info - Conditional Content */}
          <Card title="Model Information">
          <div className="space-y-4">
            {/* Accuracy - only if it exists */}
            {model.accuracy !== undefined && (
              <div>
                <p className="text-sm text-gray-400">Accuracy</p>
                <p className="text-2xl font-bold text-white">{(model.accuracy * 100).toFixed(2)}%</p>
        </div>
          )}

              {/* Trained Model Stats */}
              {isTrainedModel && model.training_samples && (
                <>
                  <div>
                    <p className="text-sm text-gray-400">Training Samples</p>
                    <p className="text-xl font-semibold text-white">
                      {model.training_samples.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">Test Samples</p>
                    <p className="text-xl font-semibold text-white">
                      {model.test_samples?.toLocaleString() || 'N/A'}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">Features</p>
                    <p className="text-xl font-semibold text-white">
                      {model.feature_count || 'N/A'}
                    </p>
                  </div>
                </>
              )}

              {/* External Model Stats */}
              {isExternalModel && (
                <>
                  <div>
                    <p className="text-sm text-gray-400">Framework</p>
                    <p className="text-xl font-semibold text-white capitalize">
                      {model.framework || 'Unknown'}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-400">Predictions Logged</p>
                    <p className="text-xl font-semibold text-white">
                      {model.predictions_logged?.toLocaleString() || '0'}
                    </p>
                  </div>
                  {model.version && (
                    <div>
                      <p className="text-sm text-gray-400">Version</p>
                      <p className="text-xl font-semibold text-white">{model.version}</p>
                    </div>
                  )}
                  <div>
                    <p className="text-sm text-gray-400">Monitoring</p>
                    <p className="text-xl font-semibold text-white">
                      {model.monitoring_enabled ? 'Enabled' : 'Disabled'}
                    </p>
                  </div>
                </>
              )}
            </div>
          </Card>

          {/* Sensitive Attributes */}
          <Card title="Sensitive Attributes">
            <div className="flex flex-wrap gap-2">
              {(model.sensitive_columns || model.sensitive_attributes || []).map((col) => (
                <span
                  key={col}
                  className="px-3 py-2 bg-yellow-600/20 text-yellow-400 rounded-lg text-sm font-medium border border-yellow-600/30"
                >
                  {col}
                </span>
              ))}
            </div>
            <p className="text-xs text-gray-500 mt-4">
              These attributes are protected from discrimination
            </p>
          </Card>

          {/* Quick Actions */}
          <Card title="Quick Actions">
            <div className="space-y-3">
              <button
                onClick={() => detectBiasMutation.mutate()}
                disabled={detectBiasMutation.isPending}
                className="w-full px-4 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 disabled:opacity-50 flex items-center justify-center gap-2 transition-colors"
              >
                {detectBiasMutation.isPending ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Running Analysis...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    {hasBiasAnalysis ? 'Re-run' : 'Run'} Bias Detection
                  </>
                )}
              </button>
              
              {hasBiasAnalysis && (
                <button
                  onClick={async () => {
                    try {
                      const token = localStorage.getItem('auth_token');
                      const response = await fetch(`http://localhost:8001/api/v1/reports/generate/${id}`, {
                        method: 'POST',
                        headers: { 
                          'Content-Type': 'application/json',
                          'Authorization': `Bearer ${token}`
                        },
                        body: JSON.stringify({
                          report_type: 'compliance',
                          include_recommendations: true,
                          format: 'pdf'
                        })
                      });
                      
                      const data = await response.json();
                      if (data.download_url) {
                        window.open(`http://localhost:8001${data.download_url}`, '_blank');
                      }
                    } catch (error) {
                      console.error('Report generation failed:', error);
                      alert('Failed to generate report');
                    }
                  }}
                  className="w-full px-4 py-3 bg-primary text-white rounded-lg hover:bg-primary flex items-center justify-center gap-2 transition-colors"
                >
                  <FileText className="w-4 h-4" />
                  Generate Report
                </button>
              )}
              
              {model.mlflow_run_id && (
                <button
                  onClick={() => window.open(`http://localhost:5000/#/runs/${model.mlflow_run_id}`, '_blank')}
                  className="w-full px-4 py-3 bg-gray-800 text-white rounded-lg hover:bg-gray-700 flex items-center justify-center gap-2 transition-colors"
                >
                  <ExternalLink className="w-4 h-4" />
                  View in MLflow
                </button>
              )}
            </div>
          </Card>
        </div>

        {/* Bias Analysis Section */}
        {biasLoading ? (
          <Card title="Bias Analysis" className="mb-6">
            <div className="text-center py-12">
              <RefreshCw className="w-8 h-8 text-indigo-400 animate-spin mx-auto mb-4" />
              <p className="text-gray-400">Loading bias analysis...</p>
            </div>
          </Card>
        ) : hasBiasAnalysis ? (
          <>
            {/* Compliance Status */}
            <div className={`mb-6 p-4 rounded-lg border ${
              biasAnalysis.compliance_status.includes('NON_COMPLIANT')
                ? 'bg-status-danger/10 border-status-danger/30'
                : biasAnalysis.compliance_status.includes('WARNING')
                ? 'bg-status-warning/10 border-status-warning/30'
                : 'bg-status-success/10 border-status-success/30'
            }`}>
              <p className="text-lg font-semibold text-white mb-1">
                Compliance Status: {biasAnalysis.compliance_status}
              </p>
              <p className="text-sm text-gray-400">
                Analyzed {new Date(biasAnalysis.analyzed_at).toLocaleString()}
              </p>
            </div>

            {/* Fairness Metrics */}
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-white mb-4">
                Fairness Metrics - {firstSensitiveCol}
              </h2>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {metrics?.disparate_impact && (
                  <MetricDisplay
                    name="Disparate Impact Ratio"
                    value={metrics.disparate_impact.ratio}
                    threshold={metrics.disparate_impact.threshold}
                    status={metrics.disparate_impact.severity.toLowerCase() as any}
                    interpretation={metrics.disparate_impact.interpretation}
                    showGauge={true}
                    gaugeMin={0}
                    gaugeMax={2}
                    details={[
                      { label: 'Group 0 Rate', value: metrics.disparate_impact.group_0_rate },
                      { label: 'Group 1 Rate', value: metrics.disparate_impact.group_1_rate },
                    ]}
                  />
                )}

                {metrics?.statistical_parity && (
                  <MetricDisplay
                    name="Statistical Parity Difference"
                    value={metrics.statistical_parity.statistical_parity_diff}
                    threshold={metrics.statistical_parity.threshold}
                    status={metrics.statistical_parity.severity.toLowerCase() as any}
                    interpretation={metrics.statistical_parity.interpretation}
                    details={[
                      { label: 'Group 0 Rate', value: metrics.statistical_parity.group_0_rate },
                      { label: 'Group 1 Rate', value: metrics.statistical_parity.group_1_rate },
                    ]}
                  />
                )}

                {metrics?.equal_opportunity && (
                  <MetricDisplay
                    name="Equal Opportunity"
                    value={metrics.equal_opportunity.difference}
                    threshold={metrics.equal_opportunity.threshold}
                    status={metrics.equal_opportunity.severity.toLowerCase() as any}
                    interpretation={metrics.equal_opportunity.interpretation}
                    details={[
                      { label: 'Group 0 TPR', value: metrics.equal_opportunity.group_0_tpr },
                      { label: 'Group 1 TPR', value: metrics.equal_opportunity.group_1_tpr },
                    ]}
                  />
                )}

                {metrics?.average_odds && (
                  <MetricDisplay
                    name="Average Odds Difference"
                    value={metrics.average_odds.average_difference}
                    threshold={metrics.average_odds.threshold}
                    status={metrics.average_odds.severity.toLowerCase() as any}
                    interpretation={metrics.average_odds.interpretation}
                    details={[
                      { label: 'Group 0 TPR', value: metrics.average_odds.group_0_tpr },
                      { label: 'Group 0 FPR', value: metrics.average_odds.group_0_fpr },
                      { label: 'Group 1 TPR', value: metrics.average_odds.group_1_tpr },
                      { label: 'Group 1 FPR', value: metrics.average_odds.group_1_fpr },
                    ]}
                  />
                )}
              </div>
            </div>

            {/* Recommendations */}
            <Card title="Recommendations" className="mb-6">
              <ul className="space-y-3">
                {biasAnalysis.recommendations.map((rec, index) => (
                  <li key={index} className="flex items-start gap-3">
                    <div className={`mt-1 w-2 h-2 rounded-full flex-shrink-0 ${
                      rec.includes('CRITICAL') ? 'bg-status-danger' :
                      rec.includes('HIGH') ? 'bg-status-warning' :
                      'bg-status-success'
                    }`} />
                    <p className="text-sm text-gray-300">{rec}</p>
                  </li>
                ))}
              </ul>
            </Card>

            {/* Mitigation Panel - ONLY for trained models */}
            {isTrainedModel && (
              <Card title="Apply Bias Mitigation" subtitle="Retrain model with fairness constraints">
                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-3">
                      Mitigation Strategy
                    </label>
                    <div className="grid grid-cols-2 gap-3">
                      {[
                        { value: 'auto', label: 'Auto', desc: 'Best strategy' },
                        { value: 'reweighing', label: 'Reweighing', desc: 'Adjust weights' },
                        { value: 'threshold_optimization', label: 'Threshold', desc: 'Optimize cutoff' },
                        { value: 'fairness_constraints', label: 'Constraints', desc: 'Train with penalties' },
                      ].map((strategy) => (
                        <button
                          key={strategy.value}
                          onClick={() => setSelectedStrategy(strategy.value)}
                          className={`p-4 rounded-lg border text-left transition-all ${
                            selectedStrategy === strategy.value
                              ? 'bg-indigo-600 border-indigo-500 text-white'
                              : 'bg-background-secondary border-gray-700 text-gray-300 hover:border-gray-600'
                          }`}
                        >
                          <p className="font-medium mb-1">{strategy.label}</p>
                          <p className="text-xs opacity-80">{strategy.desc}</p>
                        </button>
                      ))}
                    </div>
                  </div>

                  <button
                    onClick={handleMitigation}
                    disabled={mitigateMutation.isPending}
                    className="w-full px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 font-medium transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                  >
                    <Play className="w-5 h-5" />
                    Apply Mitigation
                  </button>
                </div>
              </Card>
            )}

            {/* External Model Notice */}
            {isExternalModel && (
              <div className="p-4 bg-blue-900/20 border border-blue-600/30 rounded-lg">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm text-blue-300 font-medium mb-1">
                      External Model - Mitigation Not Available
                    </p>
                    <p className="text-xs text-gray-400">
                      This model was trained outside BiasGuard. To apply bias mitigation, retrain in your original platform (SageMaker, Azure ML, etc.) with fairness constraints.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </>
        ) : (
          <Card title="Bias Analysis" className="mb-6">
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4">
                <AlertCircle className="w-8 h-8 text-gray-600" />
              </div>
              <p className="text-gray-400 mb-6">No bias analysis available yet</p>
              <button
                onClick={() => detectBiasMutation.mutate()}
                disabled={detectBiasMutation.isPending}
                className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 disabled:opacity-50 inline-flex items-center gap-2 font-medium"
              >
                {detectBiasMutation.isPending ? (
                  <>
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    Running...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Run Bias Detection
                  </>
                )}
              </button>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
};

export default ModelDetail;