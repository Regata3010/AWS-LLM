import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  FileText, 
  Download, 
  Trash2, 
  RefreshCw, 
  Filter,
  Calendar,
  CheckCircle,
  AlertTriangle,
  Clock,
  FileBarChart,
  Sparkles
} from 'lucide-react';
import { modelApi } from '../services/api';
import Card from '../components/ui/Card';

interface Report {
  filename: string;
  size_mb: number;
  created_at: string;
  download_url: string;
  model_id?: string;
}

interface GenerateReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  onGenerate: (modelId: string, reportType: string) => void;
  models: any[];
  isGenerating: boolean;
}

const GenerateReportModal: React.FC<GenerateReportModalProps> = ({
  isOpen,
  onClose,
  onGenerate,
  models,
  isGenerating
}) => {
  const [selectedModel, setSelectedModel] = useState('');
  const [reportType, setReportType] = useState('compliance');

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 z-50">
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6 max-w-md w-full">
        <h2 className="text-xl font-semibold text-white mb-4">Generate New Report</h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Select Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-indigo-500"
            >
              <option value="">Choose a model...</option>
              {models.map(model => (
                <option key={model.model_id} value={model.model_id}>
                  {model.model_type} - {model.model_id.slice(0, 12)}... ({model.bias_status})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Report Type
            </label>
            <div className="space-y-2">
              {[
                { value: 'compliance', label: 'Compliance Report', desc: 'Full CFPB/ECOA compliance analysis' },
                { value: 'executive', label: 'Executive Summary', desc: 'High-level overview for leadership' },
                { value: 'technical', label: 'Technical Report', desc: 'Detailed metrics and methodology' }
              ].map(type => (
                <button
                  key={type.value}
                  onClick={() => setReportType(type.value)}
                  className={`w-full p-3 rounded-lg border text-left transition-all ${
                    reportType === type.value
                      ? 'bg-indigo-600 border-indigo-500 text-white'
                      : 'bg-gray-800 border-gray-700 text-gray-300 hover:border-gray-600'
                  }`}
                >
                  <p className="font-medium text-sm">{type.label}</p>
                  <p className="text-xs opacity-80 mt-1">{type.desc}</p>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="flex justify-end gap-3 mt-6">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700"
            disabled={isGenerating}
          >
            Cancel
          </button>
          <button
            onClick={() => {
              if (selectedModel) {
                onGenerate(selectedModel, reportType);
              }
            }}
            disabled={!selectedModel || isGenerating}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 disabled:opacity-50 flex items-center gap-2"
          >
            {isGenerating ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                Generate Report
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

const Reports: React.FC = () => {
  const [reports, setReports] = useState<Report[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showGenerateModal, setShowGenerateModal] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [filterStatus, setFilterStatus] = useState<string>('all');

  // Fetch models for report generation
  const { data: modelsData } = useQuery({
    queryKey: ['models'],
    queryFn: () => modelApi.list({ limit: 100 }),
  });

  useEffect(() => {
    fetchReports();
  }, []);

  const fetchReports = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8001/api/v1/reports/list');
      const data = await response.json();
      setReports(data.reports || []);
    } catch (error) {
      console.error('Failed to fetch reports:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const generateReport = async (modelId: string, reportType: string) => {
    setIsGenerating(true);
    try {
      const response = await fetch(`http://localhost:8001/api/v1/reports/generate/${modelId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          report_type: reportType,
          include_recommendations: true,
          format: 'pdf'
        })
      });
      
      const data = await response.json();
      
      if (data.download_url) {
        // Refresh reports list
        await fetchReports();
        
        // Auto-download
        window.open(`http://localhost:8001${data.download_url}`, '_blank');
        
        // Close modal
        setShowGenerateModal(false);
      }
    } catch (error) {
      console.error('Failed to generate report:', error);
      alert('Report generation failed');
    } finally {
      setIsGenerating(false);
    }
  };

  const deleteReport = async (filename: string) => {
    if (!window.confirm('Are you sure you want to delete this report?')) {
      return;
    }

    try {
      await fetch(`http://localhost:8001/api/v1/reports/delete/${filename}`, {
        method: 'DELETE'
      });
      
      await fetchReports();
    } catch (error) {
      console.error('Failed to delete report:', error);
      alert('Failed to delete report');
    }
  };

  const downloadReport = (downloadUrl: string) => {
    window.open(`http://localhost:8001${downloadUrl}`, '_blank');
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - date.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  const getReportStatus = (filename: string) => {
    // Parse model status from filename if possible
    if (filename.includes('compliant')) return 'compliant';
    if (filename.includes('warning')) return 'warning';
    if (filename.includes('critical')) return 'critical';
    return 'unknown';
  };

  const filteredReports = reports.filter(report => {
    if (filterStatus === 'all') return true;
    return getReportStatus(report.filename) === filterStatus;
  });

  return (
    <div className="min-h-screen bg-background-primary">
      {/* Header */}
      <div className="px-8 py-6 border-b border-gray-800">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-white">Compliance Reports</h1>
            <p className="text-gray-400 mt-1">CFPB, ECOA, and Title VII compliance documentation</p>
          </div>
          <button
            onClick={() => setShowGenerateModal(true)}
            className="px-4 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 flex items-center gap-2 font-medium transition-colors"
          >
            <Sparkles className="w-5 h-5" />
            Generate New Report
          </button>
        </div>
      </div>

      <div className="p-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-background-card rounded-lg border border-gray-800 p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Total Reports</p>
                <p className="text-3xl font-bold text-white mt-1">{reports.length}</p>
              </div>
              <FileText className="w-8 h-8 text-indigo-400" />
            </div>
          </div>

          <div className="bg-background-card rounded-lg border border-gray-800 p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">This Week</p>
                <p className="text-3xl font-bold text-white mt-1">
                  {reports.filter(r => {
                    const date = new Date(r.created_at);
                    const weekAgo = new Date();
                    weekAgo.setDate(weekAgo.getDate() - 7);
                    return date > weekAgo;
                  }).length}
                </p>
              </div>
              <Calendar className="w-8 h-8 text-green-400" />
            </div>
          </div>

          <div className="bg-background-card rounded-lg border border-gray-800 p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Compliant</p>
                <p className="text-3xl font-bold text-white mt-1">
                  {reports.filter(r => getReportStatus(r.filename) === 'compliant').length}
                </p>
              </div>
              <CheckCircle className="w-8 h-8 text-green-500" />
            </div>
          </div>

          <div className="bg-background-card rounded-lg border border-gray-800 p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Avg Size</p>
                <p className="text-3xl font-bold text-white mt-1">
                  {reports.length > 0 
                    ? `${(reports.reduce((sum, r) => sum + r.size_mb, 0) / reports.length).toFixed(2)}`
                    : '0'
                  } <span className="text-lg text-gray-500">MB</span>
                </p>
              </div>
              <FileBarChart className="w-8 h-8 text-primary-light" />
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="mb-6 flex items-center gap-3">
          <Filter className="w-5 h-5 text-gray-400" />
          <div className="flex gap-2">
            {[
              { value: 'all', label: 'All Reports' },
              { value: 'compliant', label: 'Compliant' },
              { value: 'warning', label: 'Warning' },
              { value: 'critical', label: 'Critical' }
            ].map(filter => (
              <button
                key={filter.value}
                onClick={() => setFilterStatus(filter.value)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  filterStatus === filter.value
                    ? 'bg-indigo-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                {filter.label}
              </button>
            ))}
          </div>
          
          <button
            onClick={fetchReports}
            className="ml-auto p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
            title="Refresh reports"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
        </div>

        {/* Reports List */}
        <Card title="Generated Reports" subtitle={`${filteredReports.length} reports found`}>
          {isLoading ? (
            <div className="text-center py-12">
              <RefreshCw className="w-8 h-8 text-indigo-400 animate-spin mx-auto mb-4" />
              <p className="text-gray-400">Loading reports...</p>
            </div>
          ) : filteredReports.length === 0 ? (
            <div className="text-center py-12">
              <FileText className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-white mb-2">No Reports Yet</h3>
              <p className="text-gray-400 mb-6">
                Generate your first compliance report to get started
              </p>
              <button
                onClick={() => setShowGenerateModal(true)}
                className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 inline-flex items-center gap-2"
              >
                <Sparkles className="w-5 h-5" />
                Generate Report
              </button>
            </div>
          ) : (
            <div className="space-y-3">
              {filteredReports.map((report) => {
                const modelId = report.filename.match(/model_[a-f0-9]+/)?.[0];
                const status = getReportStatus(report.filename);
                
                return (
                  <div
                    key={report.filename}
                    className="bg-background-card border border-gray-800 rounded-lg p-5 hover:border-gray-700 transition-all"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-4 flex-1">
                        <div className="w-12 h-12 bg-indigo-600/20 rounded-lg flex items-center justify-center flex-shrink-0">
                          <FileText className="w-6 h-6 text-indigo-400" />
                        </div>
                        
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-3 mb-2">
                            <h3 className="text-white font-semibold truncate">
                              {report.filename}
                            </h3>
                            {status !== 'unknown' && (
                              <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                                status === 'compliant' 
                                  ? 'bg-green-600/20 text-green-400'
                                  : status === 'warning'
                                  ? 'bg-yellow-600/20 text-yellow-400'
                                  : 'bg-red-600/20 text-red-400'
                              }`}>
                                {status.toUpperCase()}
                              </span>
                            )}
                          </div>
                          
                          <div className="flex items-center gap-4 text-sm text-gray-400">
                            {modelId && (
                              <div className="flex items-center gap-1.5">
                                <FileBarChart className="w-3.5 h-3.5" />
                                <span className="font-mono text-xs">{modelId}</span>
                              </div>
                            )}
                            <div className="flex items-center gap-1.5">
                              <Calendar className="w-3.5 h-3.5" />
                              <span>{formatDate(report.created_at)}</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                              <Clock className="w-3.5 h-3.5" />
                              <span>{new Date(report.created_at).toLocaleTimeString()}</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                              <span>{report.size_mb.toFixed(2)} MB</span>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => downloadReport(report.download_url)}
                          className="p-2 text-indigo-400 hover:text-white hover:bg-indigo-600/20 rounded-lg transition-colors"
                          title="Download PDF"
                        >
                          <Download className="w-5 h-5" />
                        </button>
                        <button
                          onClick={() => deleteReport(report.filename)}
                          className="p-2 text-red-400 hover:text-white hover:bg-red-600/20 rounded-lg transition-colors"
                          title="Delete Report"
                        >
                          <Trash2 className="w-5 h-5" />
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </Card>

        {/* Report Templates Info */}
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-background-card border border-gray-800 rounded-lg p-5">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-indigo-600/20 rounded-lg flex items-center justify-center">
                <CheckCircle className="w-5 h-5 text-indigo-400" />
              </div>
              <h4 className="text-white font-semibold">Compliance Report</h4>
            </div>
            <p className="text-sm text-gray-400 mb-3">
              Full CFPB/ECOA compliance analysis with regulatory checklist
            </p>
            <ul className="text-xs text-gray-500 space-y-1">
              <li>• Executive summary</li>
              <li>• Fairness metrics table</li>
              <li>• Regulatory compliance checklist</li>
              <li>• LLM-powered recommendations</li>
            </ul>
          </div>

          <div className="bg-background-card border border-gray-800 rounded-lg p-5">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-primary/20 rounded-lg flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-primary-light" />
              </div>
              <h4 className="text-white font-semibold">Executive Summary</h4>
            </div>
            <p className="text-sm text-gray-400 mb-3">
              High-level overview for leadership and stakeholders
            </p>
            <ul className="text-xs text-gray-500 space-y-1">
              <li>• Business impact summary</li>
              <li>• Key findings (top 3)</li>
              <li>• Strategic recommendations</li>
              <li>• Compliance status</li>
            </ul>
          </div>

          <div className="bg-background-card border border-gray-800 rounded-lg p-5">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-green-600/20 rounded-lg flex items-center justify-center">
                <AlertTriangle className="w-5 h-5 text-green-400" />
              </div>
              <h4 className="text-white font-semibold">Technical Report</h4>
            </div>
            <p className="text-sm text-gray-400 mb-3">
              Detailed analysis for data scientists and ML engineers
            </p>
            <ul className="text-xs text-gray-500 space-y-1">
              <li>• Complete fairness metrics</li>
              <li>• Statistical methodology</li>
              <li>• Feature importance</li>
              <li>• MLflow experiment details</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Generate Report Modal */}
      <GenerateReportModal
        isOpen={showGenerateModal}
        onClose={() => setShowGenerateModal(false)}
        onGenerate={generateReport}
        models={modelsData?.models || []}
        isGenerating={isGenerating}
      />
    </div>
  );
};

export default Reports;