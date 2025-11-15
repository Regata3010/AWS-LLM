import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Shield, 
  Search, 
  Filter, 
  Trash2, 
  Eye, 
  Activity,
  Database,
  AlertCircle,
  CheckCircle,
  AlertTriangle
} from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { monitoringApi } from '../services/api';

interface ModelListItem {
  model_id: string;
  model_name: string;
  model_type: string;
  framework: string;
  version?: string;
  source: 'external' | 'trained';
  status: string;
  monitoring_enabled: boolean;
  sensitive_columns: string[];
  predictions_logged: number;
  bias_status: 'compliant' | 'warning' | 'critical' | 'unknown';
  created_at: string;
  updated_at: string;
  has_analysis: boolean;
}

const Models: React.FC = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  
  const [models, setModels] = useState<ModelListItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    setIsLoading(true);
    try {
      const response = await monitoringApi.listAllModels();
      setModels(response.models || []);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async (modelId: string) => {
    try {
      await monitoringApi.deleteExternalModel(modelId);
      setModels(models.filter(m => m.model_id !== modelId));
      setDeleteConfirm(null);
    } catch (error: any) {
      alert(`Delete failed: ${error.response?.data?.detail || error.message}`);
    }
  };

  const getBiasStatusColor = (status: string) => {
    switch (status) {
      case 'compliant': return 'bg-green-100 text-green-800 border-green-300';
      case 'warning': return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'critical': return 'bg-red-100 text-red-800 border-red-300';
      default: return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const getBiasStatusIcon = (status: string) => {
    switch (status) {
      case 'compliant': return <CheckCircle className="w-4 h-4" />;
      case 'warning': return <AlertTriangle className="w-4 h-4" />;
      case 'critical': return <AlertCircle className="w-4 h-4" />;
      default: return <Shield className="w-4 h-4" />;
    }
  };

  const filteredModels = models
    .filter(model => {
      const matchesSearch = model.model_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           model.model_id.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesStatus = statusFilter === 'all' || model.bias_status === statusFilter;
      return matchesSearch && matchesStatus;
    });

  const canDelete = user?.role === 'admin' || user?.is_superuser;

  return (
    <div className="min-h-screen bg-background-primary">
      {/* Header */}
      <div className="px-8 py-6 border-b border-gray-800">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-white">Models</h1>
            <p className="text-gray-400 mt-1">
              Manage and monitor your registered models
            </p>
          </div>
          <button
            onClick={() => navigate('/upload')}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 flex items-center gap-2"
          >
            <Shield className="w-5 h-5" />
            Register New Model
          </button>
        </div>
      </div>

      <div className="p-8">
        {/* Filters */}
        <div className="mb-6 flex gap-4">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search models by name or ID..."
              className="w-full pl-11 pr-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder:text-gray-500 focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          {/* Status Filter */}
          <div className="relative">
            <Filter className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="pl-11 pr-8 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-indigo-500 appearance-none cursor-pointer"
            >
              <option value="all">All Status</option>
              <option value="compliant">Compliant</option>
              <option value="warning">Warning</option>
              <option value="critical">Critical</option>
              <option value="unknown">Unknown</option>
            </select>
          </div>
        </div>

        {/* Results Count */}
        <p className="text-sm text-gray-400 mb-4">
          Showing {filteredModels.length} of {models.length} models
        </p>

        {/* Models List */}
        {isLoading ? (
          <div className="text-center py-12">
            <Activity className="w-12 h-12 text-indigo-500 animate-spin mx-auto mb-4" />
            <p className="text-gray-400">Loading models...</p>
          </div>
        ) : filteredModels.length === 0 ? (
          <div className="text-center py-12 bg-gray-800/50 rounded-lg border border-gray-700">
            <Database className="w-16 h-16 text-gray-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">
              {searchQuery || statusFilter !== 'all' ? 'No models match your filters' : 'No models registered yet'}
            </h3>
            <p className="text-gray-400 mb-6">
              {searchQuery || statusFilter !== 'all' ? 
                'Try adjusting your search or filters' : 
                'Register your first model to start monitoring for bias'
              }
            </p>
            {!searchQuery && statusFilter === 'all' && (
              <button
                onClick={() => navigate('/upload')}
                className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 inline-flex items-center gap-2"
              >
                <Shield className="w-5 h-5" />
                Register Your First Model
              </button>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            {filteredModels.map((model) => (
              <div
                key={model.model_id}
                className="bg-gray-800/50 border border-gray-700 rounded-lg p-6 hover:border-indigo-600/50 transition"
              >
                <div className="flex justify-between items-start mb-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="text-lg font-semibold text-white">
                        {model.model_name}
                      </h3>
                      {model.version && (
                        <span className="px-2 py-1 bg-gray-700 text-gray-300 rounded text-xs">
                          {model.version}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-gray-500 font-mono mb-3">{model.model_id}</p>
                    
                    <div className="flex flex-wrap gap-3 text-sm">
                      <span className="flex items-center gap-1 text-gray-400">
                        <Database className="w-4 h-4" />
                        {model.framework}
                      </span>
                      <span className="text-gray-400">•</span>
                      <span className="text-gray-400">
                        {model.predictions_logged.toLocaleString()} predictions
                      </span>
                      <span className="text-gray-400">•</span>
                      <span className="text-gray-400">
                        {new Date(model.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    <span className={`px-3 py-1 rounded-lg text-sm font-medium border flex items-center gap-2 ${
                      getBiasStatusColor(model.bias_status)
                    }`}>
                      {getBiasStatusIcon(model.bias_status)}
                      {model.bias_status.toUpperCase()}
                    </span>
                  </div>
                </div>

                {/* Sensitive Attributes */}
                <div className="mb-4">
                  <p className="text-xs text-gray-500 mb-2">Monitoring:</p>
                  <div className="flex flex-wrap gap-2">
                    {model.sensitive_columns.map((attr) => (
                      <span
                        key={attr}
                        className="px-2 py-1 bg-yellow-900/30 border border-yellow-600/50 text-yellow-400 rounded text-xs"
                      >
                        {attr}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-3 pt-4 border-t border-gray-700">
                  <button
                    onClick={() => navigate(`/model/${model.model_id}`)}
                    className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 flex items-center gap-2 text-sm"
                  >
                    <Eye className="w-4 h-4" />
                    View Details
                  </button>

                  {canDelete && (
                    <>
                      {deleteConfirm === model.model_id ? (
                        <div className="flex gap-2">
                          <button
                            onClick={() => handleDelete(model.model_id)}
                            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-500 text-sm"
                          >
                            Confirm Delete
                          </button>
                          <button
                            onClick={() => setDeleteConfirm(null)}
                            className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 text-sm"
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <button
                          onClick={() => setDeleteConfirm(model.model_id)}
                          className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-red-900/50 hover:text-red-400 flex items-center gap-2 text-sm transition"
                        >
                          <Trash2 className="w-4 h-4" />
                          Delete
                        </button>
                      )}
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Models;