import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { 
  ArrowRight, 
  ArrowLeft, 
  CheckCircle, 
  
  Database,
  AlertTriangle,
  Info,
  Shield,
  Activity,
  Plus,
  X
} from 'lucide-react';
import { monitoringApi } from '../services/api';
import Card from '../components/ui/Card';
import UploadZone from '../components/features/UploadZone';
import type { 
  RegisterExternalModelRegisterRequest,
  RegisterExternalModelRegisterResponse,
  AnalyzeRequest,
  AnalyzeResponse
} from '../types';
import { useAuth } from '../hooks/useAuth';

type Step = 'register' | 'upload' | 'analyze' | 'complete';

const Monitor: React.FC = () => {
  const navigate = useNavigate();

  // Wizard state
  const { user } = useAuth();
  const [currentStep, setCurrentStep] = useState<Step>('register');
  const [registeredModelId, setRegisteredModelId] = useState('');
  
  // Registration form
  const [modelName, setModelName] = useState('');
  const [modelType, setModelType] = useState('classification');
  const [framework, setFramework] = useState('sklearn');
  const [version, setVersion] = useState('');
  const [description, setDescription] = useState('');
  const [sensitiveAttributes, setSensitiveAttributes] = useState<string[]>(['race', 'gender', 'age']);
  const [newAttribute, setNewAttribute] = useState('');
  
  // Upload state
  const [uploadStats, setUploadStats] = useState<any>(null);
  const [biasAnalysis, setBiasAnalysis] = useState<AnalyzeResponse | null>(null);


  // Register model mutation
  const registerMutation = useMutation({
    mutationFn: monitoringApi.registerExternalModel,
    onSuccess: (data: RegisterExternalModelRegisterResponse) => {
      setRegisteredModelId(data.model_id);
      setCurrentStep('upload');
    },
    onError: (error: any) => {
      alert(`Registration failed: ${error.response?.data?.detail || error.message}`);
    },
  });

  // Upload CSV mutation
  const uploadMutation = useMutation({
    mutationFn: ({ modelId, file }: { modelId: string; file: File }) => 
      monitoringApi.uploadPredictionCSV(modelId, file),
    onSuccess: (data) => {
      setUploadStats(data);
      setCurrentStep('analyze');
    },
    onError: (error: any) => {
      alert(`Upload failed: ${error.response?.data?.detail || error.message}`);
    },
  });

  // Analyze bias mutation
  const analyzeMutation = useMutation({
    mutationFn: (request: AnalyzeRequest) => monitoringApi.analyzeBias(request),
    onSuccess: (data: AnalyzeResponse) => {
      setBiasAnalysis(data);
      setCurrentStep('complete');
    },
    onError: (error: any) => {
      alert(`Analysis failed: ${error.response?.data?.detail || error.message}`);
    },
  });

  const handleRegister = () => {
    if (!modelName || sensitiveAttributes.length === 0) {
      alert('Please provide model name and at least one sensitive attribute');
      return;
    }

    const request: RegisterExternalModelRegisterRequest = {
      model_name: modelName,
      model_type: modelType,
      framework,
      version: version || undefined,
      description: description || undefined,
      sensitive_attributes: sensitiveAttributes,
    };

    registerMutation.mutate(request);
  };

  const handleFileSelect = (file: File) => {
    uploadMutation.mutate({
      modelId: registeredModelId,
      file,
    });
  };

  const handleAnalyze = () => {
    const request: AnalyzeRequest = {
      model_id: registeredModelId,
    };
    analyzeMutation.mutate(request);
  };

  const addAttribute = () => {
    if (newAttribute && !sensitiveAttributes.includes(newAttribute.trim())) {
      setSensitiveAttributes([...sensitiveAttributes, newAttribute.trim()]);
      setNewAttribute('');
    }
  };

  const removeAttribute = (attr: string) => {
    setSensitiveAttributes(sensitiveAttributes.filter(a => a !== attr));
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 'register':
        return (
          <Card title="Register External Model" subtitle="Step 1 of 3 - Production Model Registration">
            <div className="space-y-6">
              {/* Info Banner */}
              <div className="p-4 bg-blue-900/20 border border-blue-600/30 rounded-lg">
                <div className="flex items-start gap-3">
                  <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm text-blue-300 font-medium mb-1">
                      Monitoring Platform - Register Your Production Model
                    </p>
                    <p className="text-xs text-gray-400">
                      BiasGuard v2.0 monitors models trained elsewhere (SageMaker, Azure ML, Databricks, etc.). 
                      Register your model, then upload prediction logs to detect bias in production.
                    </p>
                  </div>
                </div>
              </div>

              {/* Model Information */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Model Name *
                </label>
                <input
                  type="text"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder="e.g., BNPL Loan Approval Model - Production"
                  className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-indigo-500 placeholder:text-gray-500"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Model Type *
                  </label>
                  <select
                    value={modelType}
                    onChange={(e) => setModelType(e.target.value)}
                    className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="classification">Classification</option>
                    <option value="regression">Regression</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Framework
                  </label>
                  <select
                    value={framework}
                    onChange={(e) => setFramework(e.target.value)}
                    className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="sklearn">scikit-learn</option>
                    <option value="xgboost">XGBoost</option>
                    <option value="tensorflow">TensorFlow</option>
                    <option value="pytorch">PyTorch</option>
                    <option value="lightgbm">LightGBM</option>
                    <option value="catboost">CatBoost</option>
                    <option value="sagemaker">AWS SageMaker</option>
                    <option value="azureml">Azure ML</option>
                    <option value="other">Other</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Version (Optional)
                </label>
                <input
                  type="text"
                  value={version}
                  onChange={(e) => setVersion(e.target.value)}
                  placeholder="e.g., v1.0, production-2024, model-abc123"
                  className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-indigo-500 placeholder:text-gray-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Description (Optional)
                </label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Brief description of model purpose, training data, deployment environment..."
                  rows={3}
                  className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-indigo-500 placeholder:text-gray-500"
                />
              </div>

              {/* Sensitive Attributes */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Sensitive Attributes * (Protected Classes)
                </label>
                <p className="text-xs text-gray-400 mb-3">
                  These attributes will be monitored for bias. Must match column names in your prediction CSV.
                </p>
                
                <div className="flex gap-2 mb-3">
                  <input
                    type="text"
                    value={newAttribute}
                    onChange={(e) => setNewAttribute(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && addAttribute()}
                    placeholder="e.g., race, gender, age"
                    className="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-indigo-500 placeholder:text-gray-500"
                  />
                  <button
                    onClick={addAttribute}
                    className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 flex items-center gap-2"
                  >
                    <Plus className="w-4 h-4" />
                    Add
                  </button>
                </div>

                <div className="flex flex-wrap gap-2">
                  {sensitiveAttributes.map((attr) => (
                    <span
                      key={attr}
                      className="px-3 py-1.5 bg-yellow-900/30 border border-yellow-600/50 text-yellow-400 rounded-lg text-sm flex items-center gap-2"
                    >
                      {attr}
                      <button
                        onClick={() => removeAttribute(attr)}
                        className="hover:text-yellow-200"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </span>
                  ))}
                </div>

                {sensitiveAttributes.length === 0 && (
                  <p className="text-xs text-red-400 mt-2 flex items-center gap-1">
                    <AlertTriangle className="w-3 h-3" />
                    Add at least one sensitive attribute to monitor for bias
                  </p>
                )}
              </div>

              {/* Common Attributes Suggestions */}
              <div className="p-3 bg-gray-800/50 border border-gray-700 rounded-lg">
                <p className="text-xs text-gray-400 mb-2">Common sensitive attributes:</p>
                <div className="flex flex-wrap gap-2">
                  {['race', 'gender', 'age', 'ethnicity', 'religion', 'disability', 'marital_status', 'national_origin', 'sexual_orientation'].map((attr) => (
                    !sensitiveAttributes.includes(attr) && (
                      <button
                        key={attr}
                        onClick={() => setSensitiveAttributes([...sensitiveAttributes, attr])}
                        className="px-2 py-1 bg-gray-700 text-gray-300 rounded text-xs hover:bg-gray-600 transition"
                      >
                        + {attr}
                      </button>
                    )
                  ))}
                </div>
              </div>

              <button
                onClick={handleRegister}
                disabled={registerMutation.isPending || !modelName || sensitiveAttributes.length === 0}
                className="w-full px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-medium"
              >
                {registerMutation.isPending ? (
                  <>
                    <Activity className="w-5 h-5 animate-spin" />
                    Registering...
                  </>
                ) : (
                  <>
                    <Shield className="w-5 h-5" />
                    Register Model
                    <ArrowRight className="w-5 h-5" />
                  </>
                )}
              </button>
            </div>
          </Card>
        );

      case 'upload':
        return (
          <Card title="Upload Prediction Logs" subtitle="Step 2 of 3 - Upload Production Data">
            <div className="space-y-6">
              {/* Model Info Banner */}
              <div className="p-4 bg-green-900/20 border border-green-600/30 rounded-lg">
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm text-green-300 font-medium mb-1">
                      ✓ Model Registered Successfully
                    </p>
                    <p className="text-xs text-gray-400">
                      <span className="font-semibold text-white">{modelName}</span> ({framework})
                    </p>
                    <p className="text-xs text-gray-500 font-mono mt-1">
                      Model ID: {registeredModelId}
                    </p>
                  </div>
                </div>
              </div>

              {/* CSV Format Instructions */}
              <div className="p-4 bg-blue-900/20 border border-blue-600/30 rounded-lg">
                <div className="flex items-start gap-3">
                  <Database className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm text-blue-300 font-medium mb-2">
                      Required CSV Format
                    </p>
                    <div className="text-xs text-gray-400 space-y-2">
                      <p className="font-medium text-gray-300">Your CSV must contain:</p>
                      <ul className="list-disc list-inside ml-2 space-y-1">
                        <li>
                          <code className="text-yellow-400 bg-gray-800 px-1.5 py-0.5 rounded">prediction</code> 
                          <span className="ml-1">- Model's prediction (0/1 for classification, number for regression)</span>
                        </li>
                        <li>
                          <code className="text-yellow-400 bg-gray-800 px-1.5 py-0.5 rounded">ground_truth</code> 
                          <span className="ml-1">- Actual outcome (optional but recommended for accuracy metrics)</span>
                        </li>
                        {sensitiveAttributes.map(attr => (
                          <li key={attr}>
                            <code className="text-yellow-400 bg-gray-800 px-1.5 py-0.5 rounded">{attr}</code>
                            <span className="ml-1">- Sensitive attribute for bias detection</span>
                          </li>
                        ))}
                      </ul>
                      <p className="mt-2 pt-2 border-t border-blue-600/20">
                        <span className="text-gray-500">Optional:</span> 
                        <code className="text-gray-400 bg-gray-800 px-1.5 py-0.5 rounded mx-1">prediction_proba</code>
                        (confidence scores),
                        <code className="text-gray-400 bg-gray-800 px-1.5 py-0.5 rounded mx-1">features</code>
                        (JSON),
                        <code className="text-gray-400 bg-gray-800 px-1.5 py-0.5 rounded mx-1">logged_at</code>
                        (timestamp)
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <UploadZone
                onFileSelect={handleFileSelect}
                isUploading={uploadMutation.isPending}
                uploadProgress={uploadMutation.isPending ? 50 : uploadMutation.isSuccess ? 100 : 0}
              />

              {uploadMutation.isPending && (
                <div className="p-4 bg-indigo-600/10 border border-indigo-600/30 rounded-lg">
                  <div className="flex items-center gap-3">
                    <Activity className="w-5 h-5 text-indigo-400 animate-spin" />
                    <div>
                      <p className="text-white font-medium">Processing prediction logs...</p>
                      <p className="text-xs text-gray-400 mt-1">Validating format and uploading to database</p>
                    </div>
                  </div>
                </div>
              )}

              <div className="flex justify-between">
                <button
                  onClick={() => setCurrentStep('register')}
                  disabled={uploadMutation.isPending}
                  className="px-6 py-3 bg-gray-800 text-white rounded-lg hover:bg-gray-700 flex items-center gap-2 disabled:opacity-50"
                >
                  <ArrowLeft className="w-5 h-5" />
                  Back
                </button>
              </div>
            </div>
          </Card>
        );

      case 'analyze':
        return (
          <Card title="Analyze Bias" subtitle="Step 3 of 3 - Run Fairness Analysis">
            <div className="space-y-6">
              {/* Upload Success */}
              <div className="p-4 bg-green-900/20 border border-green-600/30 rounded-lg">
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm text-green-300 font-medium mb-2">
                      ✓ Upload Successful
                    </p>
                    <div className="grid grid-cols-2 gap-4 text-xs text-gray-400">
                      <div>
                        <span className="text-gray-500">Predictions Logged:</span>
                        <span className="ml-2 text-white font-semibold text-sm">
                          {uploadStats?.predictions_logged?.toLocaleString()}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-500">Batch ID:</span>
                        <span className="ml-2 text-white font-mono text-[10px]">
                          {uploadStats?.batch_id}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Upload Statistics */}
              {uploadStats?.statistics && (
                <div className="p-4 bg-gray-800/50 border border-gray-700 rounded-lg">
                  <h4 className="text-white font-medium mb-3 flex items-center gap-2">
                    <Database className="w-4 h-4" />
                    Upload Statistics
                  </h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Total Predictions:</span>
                      <span className="text-white font-semibold">
                        {uploadStats.statistics.total_predictions?.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Overall Approval Rate:</span>
                      <span className="text-white font-semibold">
                        {(uploadStats.statistics.overall_approval_rate * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Has Ground Truth:</span>
                      <span className={uploadStats.statistics.has_ground_truth ? 'text-green-400' : 'text-yellow-400'}>
                        {uploadStats.statistics.has_ground_truth ? '✓ Yes' : '⚠ No'}
                      </span>
                    </div>
                    {uploadStats.statistics.date_range && (
                      <div className="flex justify-between">
                        <span className="text-gray-400">Date Range:</span>
                        <span className="text-white text-xs">
                          {new Date(uploadStats.statistics.date_range.earliest).toLocaleDateString()} - 
                          {new Date(uploadStats.statistics.date_range.latest).toLocaleDateString()}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Breakdown by Attributes */}
              {uploadStats?.statistics?.breakdown_by_attribute && (
                <div className="p-4 bg-gray-800/50 border border-gray-700 rounded-lg max-h-96 overflow-y-auto">
                  <h4 className="text-white font-medium mb-3">Approval Rates by Group</h4>
                  {Object.entries(uploadStats.statistics.breakdown_by_attribute).map(([attr, groups]: [string, any]) => (
                    <div key={attr} className="mb-4 last:mb-0">
                      <p className="text-sm text-indigo-400 mb-2 font-medium capitalize">{attr}:</p>
                      <div className="space-y-1.5">
                        {Object.entries(groups).slice(0, 10).map(([group, data]: [string, any]) => (
                          <div key={group} className="flex justify-between items-center text-xs bg-gray-900/50 px-3 py-2 rounded">
                            <span className="text-gray-300">{group}</span>
                            <div className="flex items-center gap-3">
                              <span className="text-gray-500">{data.count} samples</span>
                              <span className="text-white font-medium">
                                {(data.approval_rate * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Analyze Button */}
              <button
                onClick={handleAnalyze}
                disabled={analyzeMutation.isPending}
                className="w-full px-6 py-4 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 disabled:opacity-50 flex items-center justify-center gap-2 text-lg font-semibold"
              >
                {analyzeMutation.isPending ? (
                  <>
                    <Activity className="w-6 h-6 animate-spin" />
                    Analyzing Bias...
                  </>
                ) : (
                  <>
                    <Shield className="w-6 h-6" />
                    Run Bias Analysis
                    <ArrowRight className="w-5 h-5" />
                  </>
                )}
              </button>

              <div className="flex justify-between">
                <button
                  onClick={() => setCurrentStep('upload')}
                  disabled={analyzeMutation.isPending}
                  className="px-6 py-3 bg-gray-800 text-white rounded-lg hover:bg-gray-700 flex items-center gap-2 disabled:opacity-50"
                >
                  <ArrowLeft className="w-5 h-5" />
                  Back
                </button>
              </div>
            </div>
          </Card>
        );

      case 'complete':
        return (
          <Card title="Analysis Complete!" subtitle="Bias Detection Results">
            <div className="space-y-6">
              <div className="text-center py-8">
                <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6 ${
                  biasAnalysis?.bias_status === 'compliant' 
                    ? 'bg-green-500/20' 
                    : biasAnalysis?.bias_status === 'warning'
                    ? 'bg-yellow-500/20'
                    : 'bg-red-500/20'
                }`}>
                  {biasAnalysis?.bias_status === 'compliant' ? (
                    <CheckCircle className="w-12 h-12 text-green-400" />
                  ) : (
                    <AlertTriangle className="w-12 h-12 text-yellow-400" />
                  )}
                </div>
                
                <h2 className="text-2xl font-bold text-white mb-2">
                  Bias Analysis Complete
                </h2>
                <p className={`text-lg font-semibold mb-4 uppercase ${
                  biasAnalysis?.bias_status === 'compliant' 
                    ? 'text-green-400' 
                    : biasAnalysis?.bias_status === 'warning'
                    ? 'text-yellow-400'
                    : 'text-red-400'
                }`}>
                  Status: {biasAnalysis?.bias_status}
                </p>
                <p className="text-gray-400 mb-2">{modelName}</p>
                <p className="text-xs text-gray-500 font-mono mb-8">{registeredModelId}</p>

                {/* Quick Metrics */}
                {biasAnalysis?.fairness_metrics && (
                  <div className="grid grid-cols-2 gap-4 mb-8 max-w-lg mx-auto">
                    <div className="p-4 bg-gray-800 rounded-lg border border-gray-700">
                      <p className="text-xs text-gray-400 mb-1">Disparate Impact</p>
                      <p className="text-2xl font-bold text-white">
                        {Object.values(biasAnalysis.fairness_metrics)[0]?.disparate_impact?.ratio || 'N/A'}
                      </p>
                    </div>
                    <div className="p-4 bg-gray-800 rounded-lg border border-gray-700">
                      <p className="text-xs text-gray-400 mb-1">Statistical Parity</p>
                      <p className="text-2xl font-bold text-white">
                        {Object.values(biasAnalysis.fairness_metrics)[0]?.statistical_parity?.statistical_parity_diff?.toFixed(3) || 'N/A'}
                      </p>
                    </div>
                  </div>
                )}

                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <button
                    onClick={() => navigate(`/model/${registeredModelId}`)}
                    className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 font-medium"
                  >
                    View Full Analysis
                  </button>
                  <button
                    onClick={() => navigate('/monitor')}
                    className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-500 font-medium"
                  >
                    Real-Time Monitoring
                  </button>
                  <button
                    onClick={() => {
                      setCurrentStep('register');
                      setRegisteredModelId('');
                      setModelName('');
                      setUploadStats(null);
                      setBiasAnalysis(null);
                    }}
                    className="px-6 py-3 bg-gray-800 text-white rounded-lg hover:bg-gray-700 font-medium"
                  >
                    Monitor Another Model
                  </button>
                </div>
              </div>
            </div>
          </Card>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-background-primary">
      <div className="px-8 py-6 border-b border-gray-800">
        <h1 className="text-3xl font-bold text-white">Monitor Production Models</h1>
        <p className="text-gray-400 mt-1">
          Register external models and upload prediction logs for bias monitoring
        </p>
      </div>

      {user && user.role === 'member' && (
      <div className="px-8 py-3 bg-blue-900/20 border-b border-blue-600/30">
        <div className="flex items-center gap-2 text-sm text-blue-300">
          <Info className="w-4 h-4" />
          <span>
            You're a <strong>Member</strong>. You can create and monitor models, but cannot delete them or invite users.
          </span>
        </div>
      </div>
      )}
 
      <div className="p-8">
        <div className="max-w-4xl mx-auto">
          {/* Progress Steps */}
          {currentStep !== 'complete' && (
            <div className="mb-8">
              <div className="flex items-center justify-between">
                {['Register', 'Upload', 'Analyze'].map((label, index) => {
                  const stepValues: Step[] = ['register', 'upload', 'analyze'];
                  const currentIndex = stepValues.indexOf(currentStep);
                  const isActive = index === currentIndex;
                  const isComplete = index < currentIndex;

                  return (
                    <React.Fragment key={label}>
                      <div className="flex flex-col items-center">
                        <div className={`w-10 h-10 rounded-full flex items-center justify-center border-2 ${
                            isComplete ? 'bg-indigo-600 border-indigo-600 text-white' :
                            isActive ? 'border-indigo-600 text-indigo-400 bg-indigo-600/10' :
                            'border-gray-700 text-gray-600 bg-gray-800'
                          }`}
                        >
                          {isComplete ? <CheckCircle className="w-5 h-5" /> : <span className="font-semibold">{index + 1}</span>}
                        </div>
                        <span className={`text-xs mt-2 ${isActive || isComplete ? 'text-white' : 'text-gray-500'}`}>
                          {label}
                        </span>
                      </div>

                      {index < 2 && (
                        <div className="flex-1 h-0.5 bg-gray-800 mx-4 mt-5">
                          <div className="h-full bg-indigo-600 transition-all" style={{ width: isComplete ? '100%' : '0%' }} />
                        </div>
                      )}
                    </React.Fragment>
                  );
                })}
              </div>
            </div>
          )}

          {renderStepContent()}
        </div>
      </div>
    </div>
  );
};

export default Monitor;