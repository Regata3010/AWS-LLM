import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  AlertTriangle, 
  Shield, 
  FileText, 
  RefreshCw, 
  Zap, 
  Server,
  TrendingUp,
  TrendingDown,
  Minus,
  AlertCircle,
  CheckCircle,
  XCircle,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import type { 
  Model, 
  DashboardData, 
  EnhancedModelMetrics 
} from '../types';

const RealTimeMonitor: React.FC = () => {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [modelMetrics, setModelMetrics] = useState<EnhancedModelMetrics | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('connecting');
  const [modelsList, setModelsList] = useState<Model[]>([]);
  const [reportGenerating, setReportGenerating] = useState<string | null>(null);
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    drift: true,
    trends: true,
    performance: true,
    alerts: true
  });
  
  const dashboardWsRef = useRef<WebSocket | null>(null);
  const modelWsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    fetchModels();
    connectDashboardWebSocket();
    
    return () => {
      if (dashboardWsRef.current) {
        dashboardWsRef.current.close();
      }
      if (modelWsRef.current) {
        modelWsRef.current.close();
      }
    };
  }, []);

  const fetchModels = async () => {
    try {
      const token = localStorage.getItem('auth_token');
      const response = await fetch('http://localhost:8001/api/v1/models', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      const data = await response.json();
      setModelsList(data.models || []);
      
      if (data.models && data.models.length > 0 && !selectedModel) {
        selectModel(data.models[0].model_id);
      }
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  };

  const connectDashboardWebSocket = () => {
    try {
      const token = localStorage.getItem('auth_token');
      if (!token) {
        console.error('No auth token found for WebSocket connection');
        setConnectionStatus('error');
        return;
      }
      
      const ws = new WebSocket(`ws://localhost:8001/ws/dashboard?token=${token}`);
      dashboardWsRef.current = ws;
      
      ws.onopen = () => {
        console.log('Dashboard WebSocket connected');
        setConnectionStatus('connected');
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('Dashboard update:', data);
          setDashboardData(data);
        } catch (error) {
          console.error('Failed to parse dashboard data:', error);
        }
      };
      
      ws.onerror = (error) => {
        console.error('Dashboard WebSocket error:', error);
        setConnectionStatus('error');
      };
      
      ws.onclose = () => {
        console.log('Dashboard WebSocket closed');
        setConnectionStatus('disconnected');
        
        setTimeout(() => {
          if (dashboardWsRef.current?.readyState === WebSocket.CLOSED) {
            console.log('Reconnecting dashboard WebSocket...');
            connectDashboardWebSocket();
          }
        }, 3000);
      };
    } catch (error) {
      console.error('Failed to connect dashboard WebSocket:', error);
      setConnectionStatus('error');
    }
  };

  const selectModel = (modelId: string) => {
    console.log(`Selecting model: ${modelId}`);
    setSelectedModel(modelId);
    
    if (modelWsRef.current) {
      modelWsRef.current.close();
    }
    
    try {
      const token = localStorage.getItem('auth_token');
      if (!token) {
        console.error('No auth token found for model WebSocket');
        return;
      }
      
      const ws = new WebSocket(`ws://localhost:8001/ws/monitor/${modelId}?token=${token}`);
      modelWsRef.current = ws;
      
      ws.onopen = () => {
        console.log(`Connected to model ${modelId} WebSocket`);
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('Enhanced metrics update:', data);
          setModelMetrics(data);
        } catch (error) {
          console.error('Failed to parse model metrics:', error);
        }
      };
      
      ws.onerror = (error) => {
        console.error('Model WebSocket error:', error);
      };
      
      ws.onclose = () => {
        console.log(`Model ${modelId} WebSocket closed`);
      };
    } catch (error) {
      console.error('Failed to connect model WebSocket:', error);
    }
  };

  const generateReport = async (modelId: string) => {
    setReportGenerating(modelId);
    try {
      const token = localStorage.getItem('auth_token');
      const response = await fetch(`http://localhost:8001/api/v1/reports/generate/${modelId}`, {
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
      console.error('Failed to generate report:', error);
      alert('Report generation failed');
    } finally {
      setReportGenerating(null);
    }
  };

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const getStatusColor = (status: string): string => {
    const colors: Record<string, string> = {
      'connected': 'bg-green-100 text-green-800',
      'disconnected': 'bg-red-100 text-red-800',
      'connecting': 'bg-yellow-100 text-yellow-800',
      'error': 'bg-red-100 text-red-800'
    };
    return colors[status] || colors['connecting'];
  };

  const getBiasStatusColor = (status: string): string => {
    switch(status) {
      case 'compliant': return 'bg-green-100 text-green-800';
      case 'warning': return 'bg-yellow-100 text-yellow-800';
      case 'critical': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getDriftStatusColor = (status: string): string => {
    switch(status) {
      case 'stable': return 'bg-green-500';
      case 'warning': return 'bg-yellow-500';
      case 'critical': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getTrendIcon = (direction: string) => {
    switch(direction) {
      case 'improving': return <TrendingDown className="w-4 h-4 text-green-500" />;
      case 'degrading': return <TrendingUp className="w-4 h-4 text-red-500" />;
      case 'stable': return <Minus className="w-4 h-4 text-gray-500" />;
      default: return <Minus className="w-4 h-4 text-gray-500" />;
    }
  };

  const getTrendColor = (direction: string): string => {
    switch(direction) {
      case 'improving': return 'text-green-600 bg-green-50';
      case 'degrading': return 'text-red-600 bg-red-50';
      case 'stable': return 'text-gray-600 bg-gray-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getAlertIcon = (type: string) => {
    switch(type) {
      case 'critical': return <XCircle className="w-5 h-5 text-red-600" />;
      case 'warning': return <AlertTriangle className="w-5 h-5 text-yellow-600" />;
      case 'info': return <AlertCircle className="w-5 h-5 text-blue-600" />;
      default: return <AlertCircle className="w-5 h-5 text-gray-600" />;
    }
  };

  const formatMetricName = (name: string): string => {
    return name.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  return (
    <div className="min-h-screen bg-background-primary">
      <div className="bg-card-background border-b border-border px-6 py-4">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-text-primary">Real-Time Monitoring</h1>
            <p className="text-text-secondary text-sm mt-1">
              Enhanced bias detection with drift analysis, trends, and performance tracking
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            <button 
              onClick={fetchModels}
              className="p-2 hover:bg-background-primary rounded-lg transition"
              title="Refresh models"
            >
              <RefreshCw className="w-5 h-5 text-text-secondary" />
            </button>
            
            <div className={`px-4 py-2 rounded-lg font-medium flex items-center gap-2 ${getStatusColor(connectionStatus)}`}>
              <Activity className="w-4 h-4" />
              {connectionStatus === 'connected' ? 'Live' : connectionStatus}
            </div>
          </div>
        </div>
      </div>

      <div className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-card-background rounded-lg p-6 border border-border">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-sm text-text-secondary">Total Models</p>
                <p className="text-3xl font-bold text-text-primary mt-1">
                  {dashboardData?.total_models || modelsList.length || 0}
                </p>
              </div>
              <Shield className="w-8 h-8 text-primary" />
            </div>
          </div>

          <div className="bg-card-background rounded-lg p-6 border border-border">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-sm text-text-secondary">Models at Risk</p>
                <p className="text-3xl font-bold text-orange-500 mt-1">
                  {dashboardData?.models_at_risk || 0}
                </p>
              </div>
              <AlertTriangle className="w-8 h-8 text-orange-500" />
            </div>
          </div>

          <div className="bg-card-background rounded-lg p-6 border border-border">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-sm text-text-secondary">Predictions/sec</p>
                <p className="text-3xl font-bold text-text-primary mt-1">
                  {dashboardData?.system_health?.predictions_per_second || 0}
                </p>
              </div>
              <Zap className="w-8 h-8 text-green-500" />
            </div>
          </div>

          <div className="bg-card-background rounded-lg p-6 border border-border">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-sm text-text-secondary">API Latency</p>
                <p className="text-3xl font-bold text-text-primary mt-1">
                  {dashboardData?.system_health?.api_latency_ms || 0}ms
                </p>
              </div>
              <Server className="w-8 h-8 text-primary" />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 bg-card-background rounded-lg border border-border">
            <div className="p-6 border-b border-border">
              <h2 className="text-xl font-semibold text-text-primary">Models</h2>
            </div>
            
            <div className="p-4">
              <div className="space-y-2 max-h-[700px] overflow-y-auto">
                {modelsList.map((model) => (
                  <div
                    key={model.model_id}
                    onClick={() => selectModel(model.model_id)}
                    className={`border rounded-lg p-3 cursor-pointer transition-all hover:shadow-md ${
                      selectedModel === model.model_id 
                        ? 'border-primary bg-primary/5' 
                        : 'border-border hover:border-primary/50'
                    }`}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex-1">
                        <h3 className="font-medium text-text-primary text-sm">{model.model_type}</h3>
                        <p className="text-xs text-text-secondary mt-1 font-mono">
                          {model.model_id.slice(0, 16)}...
                        </p>
                      </div>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        getBiasStatusColor(model.bias_status)
                      }`}>
                        {model.bias_status?.toUpperCase()}
                      </span>
                    </div>
                    <div className="text-xs text-text-secondary">
                      Accuracy: {model.accuracy ? (model.accuracy * 100).toFixed(1) : 'N/A'}%
                    </div>
                  </div>
                ))}
                
                {modelsList.length === 0 && (
                  <div className="text-center py-12 text-text-secondary">
                    <Shield className="w-12 h-12 mx-auto mb-3 opacity-30" />
                    <p className="text-sm">No models found</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="lg:col-span-2 space-y-4">
            {modelMetrics && selectedModel ? (
              <>
                <div className="bg-card-background rounded-lg border border-border p-4">
                  <div className="flex justify-between items-start">
                    <div>
                      <h2 className="text-lg font-semibold text-text-primary">{modelMetrics.model_name}</h2>
                      <p className="text-xs text-text-secondary font-mono mt-1">{modelMetrics.model_id}</p>
                    </div>
                    <button
                      onClick={() => generateReport(selectedModel)}
                      disabled={reportGenerating === selectedModel}
                      className="px-3 py-2 bg-primary text-white rounded-lg hover:bg-primary-hover transition disabled:opacity-50 flex items-center gap-2"
                    >
                      {reportGenerating === selectedModel ? (
                        <RefreshCw className="w-4 h-4 animate-spin" />
                      ) : (
                        <FileText className="w-4 h-4" />
                      )}
                      <span className="text-sm">Report</span>
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-4 mt-4 text-sm">
                    <div>
                      <span className="text-text-secondary">Predictions/hr:</span>
                      <span className="ml-2 font-semibold text-text-primary">
                        {modelMetrics.predictions_last_hour?.toLocaleString()}
                      </span>
                    </div>
                    <div>
                      <span className="text-text-secondary">Avg Confidence:</span>
                      <span className="ml-2 font-semibold text-text-primary">
                        {(modelMetrics.avg_confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-card-background rounded-lg border border-border">
                  <div 
                    className="p-4 border-b border-border flex justify-between items-center cursor-pointer hover:bg-background-primary/50"
                    onClick={() => toggleSection('drift')}
                  >
                    <h3 className="text-lg font-semibold text-text-primary">Drift Analysis</h3>
                    {expandedSections.drift ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                  </div>
                  
                  {expandedSections.drift && (
                    <div className="p-4 space-y-4">
                      <div className="flex justify-between items-center">
                        <div>
                          <p className="text-3xl font-bold text-text-primary">
                            {modelMetrics.drift_status?.drift_score?.toFixed(4) || '0.0000'}
                          </p>
                          <p className="text-sm text-text-secondary mt-1">Drift Score</p>
                        </div>
                        <span className={`px-4 py-2 rounded-full text-sm font-medium ${
                          modelMetrics.drift_status?.status === 'stable' 
                            ? 'bg-green-100 text-green-800'
                            : modelMetrics.drift_status?.status === 'warning'
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {modelMetrics.drift_status?.status?.toUpperCase()}
                        </span>
                      </div>

                      <div>
                        <div className="flex justify-between text-xs text-text-secondary mb-1">
                          <span>Stable</span>
                          <span>Warning (0.10)</span>
                          <span>Critical (0.15)</span>
                        </div>
                        <div className="h-3 bg-background-primary rounded-full overflow-hidden relative">
                          <div 
                            className={`h-full transition-all ${
                              getDriftStatusColor(modelMetrics.drift_status?.status || 'stable')
                            }`}
                            style={{
                              width: `${Math.min((modelMetrics.drift_status?.drift_score || 0) * 500, 100)}%`
                            }}
                          />
                          <div className="absolute left-[50%] top-0 bottom-0 w-px bg-yellow-400/30" />
                          <div className="absolute left-[75%] top-0 bottom-0 w-px bg-red-400/30" />
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4 pt-2 border-t border-border">
                        <div className="bg-background-primary rounded-lg p-3">
                          <p className="text-xs text-text-secondary mb-1">Drift Velocity</p>
                          <p className="text-xl font-bold text-text-primary">
                            {modelMetrics.drift_status?.velocity?.toFixed(4) || '0.0000'}
                          </p>
                          <p className="text-xs text-text-secondary mt-1">per analysis</p>
                        </div>
                        <div className="bg-background-primary rounded-lg p-3">
                          <p className="text-xs text-text-secondary mb-1">Trend</p>
                          <div className="flex items-center gap-2 mt-1">
                            {modelMetrics.drift_status?.trend === 'increasing' && (
                              <TrendingUp className="w-5 h-5 text-red-500" />
                            )}
                            {modelMetrics.drift_status?.trend === 'decreasing' && (
                              <TrendingDown className="w-5 h-5 text-green-500" />
                            )}
                            {modelMetrics.drift_status?.trend === 'stable' && (
                              <Minus className="w-5 h-5 text-gray-500" />
                            )}
                            <span className="text-sm font-semibold capitalize">
                              {modelMetrics.drift_status?.trend}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {modelMetrics.trends && Object.keys(modelMetrics.trends).length > 0 && (
                  <div className="bg-card-background rounded-lg border border-border">
                    <div 
                      className="p-4 border-b border-border flex justify-between items-center cursor-pointer hover:bg-background-primary/50"
                      onClick={() => toggleSection('trends')}
                    >
                      <h3 className="text-lg font-semibold text-text-primary">Fairness Trends</h3>
                      {expandedSections.trends ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                    </div>
                    
                    {expandedSections.trends && (
                      <div className="p-4 space-y-3">
                        {Object.entries(modelMetrics.trends).map(([metric, data]) => (
                          <div key={metric} className="bg-background-primary rounded-lg p-3">
                            <div className="flex justify-between items-center mb-2">
                              <span className="text-sm font-medium text-text-primary">
                                {formatMetricName(metric)}
                              </span>
                              <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${
                                getTrendColor(data.direction)
                              }`}>
                                {getTrendIcon(data.direction)}
                                <span className="capitalize">{data.direction}</span>
                              </div>
                            </div>
                            <div className="grid grid-cols-3 gap-2 text-xs">
                              <div>
                                <p className="text-text-secondary">Recent</p>
                                <p className="font-semibold text-text-primary">
                                  {data.recent_avg.toFixed(3)}
                                </p>
                              </div>
                              <div>
                                <p className="text-text-secondary">Baseline</p>
                                <p className="font-semibold text-text-primary">
                                  {data.baseline_avg.toFixed(3)}
                                </p>
                              </div>
                              <div>
                                <p className="text-text-secondary">Change</p>
                                <p className={`font-semibold ${
                                  data.change_rate > 0 ? 'text-red-600' : 'text-green-600'
                                }`}>
                                  {data.change_rate > 0 ? '+' : ''}{data.change_rate.toFixed(3)}
                                </p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {modelMetrics.performance && (
                  <div className="bg-card-background rounded-lg border border-border">
                    <div 
                      className="p-4 border-b border-border flex justify-between items-center cursor-pointer hover:bg-background-primary/50"
                      onClick={() => toggleSection('performance')}
                    >
                      <h3 className="text-lg font-semibold text-text-primary">Performance Tracking</h3>
                      {expandedSections.performance ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                    </div>
                    
                    {expandedSections.performance && (
                      <div className="p-4">
                        <div className="grid grid-cols-3 gap-4">
                          <div className="bg-background-primary rounded-lg p-3 text-center">
                            <p className="text-xs text-text-secondary mb-1">Baseline</p>
                            <p className="text-2xl font-bold text-text-primary">
                              {(modelMetrics.performance.baseline_accuracy * 100).toFixed(1)}%
                            </p>
                          </div>
                          <div className="bg-background-primary rounded-lg p-3 text-center">
                            <p className="text-xs text-text-secondary mb-1">Current</p>
                            <p className="text-2xl font-bold text-text-primary">
                              {(modelMetrics.performance.current_accuracy * 100).toFixed(1)}%
                            </p>
                          </div>
                          <div className="bg-background-primary rounded-lg p-3 text-center">
                            <p className="text-xs text-text-secondary mb-1">Drop</p>
                            <p className={`text-2xl font-bold ${
                              modelMetrics.performance.accuracy_drop > 0.02 
                                ? 'text-red-600' 
                                : 'text-green-600'
                            }`}>
                              {(modelMetrics.performance.accuracy_drop * 100).toFixed(1)}%
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                <div className="bg-card-background rounded-lg border border-border">
                  <div 
                    className="p-4 border-b border-border flex justify-between items-center cursor-pointer hover:bg-background-primary/50"
                    onClick={() => toggleSection('alerts')}
                  >
                    <div className="flex items-center gap-2">
                      <h3 className="text-lg font-semibold text-text-primary">Alerts</h3>
                      {modelMetrics.alerts && modelMetrics.alerts.length > 0 && (
                        <span className="px-2 py-1 bg-red-100 text-red-800 rounded-full text-xs font-medium">
                          {modelMetrics.alerts.length}
                        </span>
                      )}
                    </div>
                    {expandedSections.alerts ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                  </div>
                  
                  {expandedSections.alerts && (
                    <div className="p-4">
                      {modelMetrics.alerts && modelMetrics.alerts.length > 0 ? (
                        <div className="space-y-3">
                          {modelMetrics.alerts.map((alert, idx) => (
                            <div 
                              key={idx}
                              className={`p-4 rounded-lg border ${
                                alert.type === 'critical'
                                  ? 'bg-red-50 border-red-200'
                                  : alert.type === 'warning'
                                  ? 'bg-yellow-50 border-yellow-200'
                                  : 'bg-blue-50 border-blue-200'
                              }`}
                            >
                              <div className="flex items-start gap-3">
                                {getAlertIcon(alert.type)}
                                <div className="flex-1">
                                  <div className="flex justify-between items-start mb-1">
                                    <p className="font-semibold text-sm capitalize">
                                      {formatMetricName(alert.metric)}
                                      {alert.attribute && ` - ${alert.attribute}`}
                                    </p>
                                    {alert.regulation && (
                                      <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs font-medium">
                                        {alert.regulation}
                                      </span>
                                    )}
                                  </div>
                                  <p className="text-sm text-gray-700 mb-2">{alert.message}</p>
                                  <div className="flex gap-4 text-xs text-gray-600">
                                    <span>Value: <strong>{alert.value?.toFixed(3)}</strong></span>
                                    <span>Threshold: <strong>{alert.threshold}</strong></span>
                                  </div>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-center py-8 bg-green-50 rounded-lg border border-green-200">
                          <CheckCircle className="w-12 h-12 text-green-600 mx-auto mb-2" />
                          <p className="text-green-700 font-medium">No Active Alerts</p>
                          <p className="text-sm text-green-600 mt-1">All metrics within acceptable ranges</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {modelMetrics.current_bias_metrics && (
                  <div className="bg-card-background rounded-lg border border-border p-4">
                    <h3 className="text-lg font-semibold text-text-primary mb-4">Current Metrics</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-background-primary rounded-lg p-3">
                        <p className="text-xs text-text-secondary mb-1">Disparate Impact</p>
                        <p className="text-xl font-bold text-text-primary">
                          {modelMetrics.current_bias_metrics.disparate_impact?.toFixed(3)}
                        </p>
                      </div>
                      <div className="bg-background-primary rounded-lg p-3">
                        <p className="text-xs text-text-secondary mb-1">Statistical Parity</p>
                        <p className="text-xl font-bold text-text-primary">
                          {modelMetrics.current_bias_metrics.statistical_parity?.toFixed(3)}
                        </p>
                      </div>
                      <div className="bg-background-primary rounded-lg p-3">
                        <p className="text-xs text-text-secondary mb-1">Equal Opportunity</p>
                        <p className="text-xl font-bold text-text-primary">
                          {modelMetrics.current_bias_metrics.equal_opportunity?.toFixed(3)}
                        </p>
                      </div>
                      <div className="bg-background-primary rounded-lg p-3">
                        <p className="text-xs text-text-secondary mb-1">Accuracy</p>
                        <p className="text-xl font-bold text-text-primary">
                          {(modelMetrics.current_bias_metrics.accuracy * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                <div className="text-center text-xs text-text-secondary">
                  Last updated: {new Date(modelMetrics.timestamp).toLocaleString()}
                </div>
              </>
            ) : (
              <div className="bg-card-background rounded-lg border border-border p-12 text-center">
                <Activity className="w-16 h-16 mx-auto mb-4 text-text-secondary opacity-30" />
                <p className="text-text-secondary text-lg">Select a model to view enhanced metrics</p>
                <p className="text-text-secondary text-sm mt-2">
                  Real-time drift analysis, trends, and performance tracking
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RealTimeMonitor;