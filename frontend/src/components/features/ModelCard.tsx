import React from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Calendar, 
  TrendingUp, 
  AlertCircle, 
  Sparkles, 
  Link as LinkIcon,
  Activity
} from 'lucide-react';
import Badge, { getBiasStatusVariant } from '../ui/Badge';
import type { Model } from '../../types';

interface ModelCardProps {
  model: Model;
}

const ModelCard: React.FC<ModelCardProps> = ({ model }) => {
  const navigate = useNavigate();
  
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

  // Check if this is a mitigated model
  const isMitigated = model.is_mitigated || model.parent_model_id;
  
  // Check model source
  const isExternal = model.source === 'external';

  return (
    <div
      onClick={() => navigate(`/model/${model.model_id}`)}
      className={`bg-background-card border rounded-lg p-5 hover:shadow-lg transition-all cursor-pointer group relative ${
        isMitigated 
          ? 'border-primary/50 hover:border-primary' 
          : 'border-gray-800 hover:border-gray-700'
      }`}
    >
      {/* Mitigation Badge */}
      {isMitigated && (
        <div className="absolute -top-2 -right-2">
          <div className="bg-purple-600 rounded-full p-1.5 shadow-lg">
            <Sparkles className="w-3 h-3 text-white" />
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-white group-hover:text-indigo-400 transition-colors">
            {isExternal ? model.model_name : model.model_id}
          </h3>
          <p className="text-sm text-gray-400 mt-1">
            {model.model_type} {model.task_type && `• ${model.task_type}`}
          </p>
          
          {/* Source badge */}
          <div className="mt-2 flex items-center gap-2">
            <span className={`px-2 py-0.5 rounded text-xs font-medium border ${
              isExternal 
                ? 'bg-blue-600/20 text-blue-400 border-blue-600/30'
                : 'bg-green-600/20 text-green-400 border-green-600/30'
            }`}>
              {isExternal ? 'External' : 'Trained'}
            </span>
            
            {/* Mitigation info */}
            {isMitigated && (
              <>
                <span className="px-2 py-0.5 bg-primary/20 text-primary-light rounded text-xs font-medium border border-primary/30">
                  Mitigated
                </span>
                {model.mitigation_strategy && (
                  <span className="text-xs text-gray-500">
                    via {model.mitigation_strategy.replace('_', ' ')}
                  </span>
                )}
              </>
            )}
          </div>
        </div>
        <Badge variant={getBiasStatusVariant(model.bias_status)} size="sm">
          {model.bias_status}
        </Badge>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        {/* Accuracy (both trained and external) */}
        {model.accuracy !== undefined && (
          <div>
            <p className="text-xs text-gray-500 mb-1">Accuracy</p>
            <p className="text-2xl font-bold text-white">
              {(model.accuracy * 100).toFixed(2)}%
            </p>
          </div>
        )}
        
        {/* Test Samples (trained models only) */}
        {!isExternal && model.test_samples !== undefined && (
          <div>
            <p className="text-xs text-gray-500 mb-1">Test Samples</p>
            <p className="text-2xl font-bold text-white">
              {model.test_samples.toLocaleString()}
            </p>
          </div>
        )}
        
        {/* Predictions Logged (external models only) */}
        {isExternal && (
          <div>
            <p className="text-xs text-gray-500 mb-1">Predictions Logged</p>
            <p className="text-2xl font-bold text-white">
              {model.predictions_logged?.toLocaleString() ?? 'N/A'}
            </p>
          </div>
        )}
      </div>

      {/* Sensitive Attributes */}
      <div className="mb-4">
        <p className="text-xs text-gray-500 mb-2">Sensitive Attributes</p>
        <div className="flex flex-wrap gap-1">
          {(model.sensitive_columns || model.sensitive_attributes || []).slice(0, 3).map((col) => (
            <span
              key={col}
              className="px-2 py-1 bg-gray-800 text-gray-300 rounded text-xs"
            >
              {col}
            </span>
          ))}
          {(model.sensitive_columns || model.sensitive_attributes || []).length > 3 && (
            <span className="px-2 py-1 bg-gray-800 text-gray-400 rounded text-xs">
              +{(model.sensitive_columns || model.sensitive_attributes || []).length - 3} more
            </span>
          )}
        </div>
      </div>

      {/* Parent Model Link (if mitigated) */}
      {isMitigated && model.parent_model_id && (
        <div className="mb-4 p-2 bg-primary/10 border border-primary/30 rounded">
          <div className="flex items-center gap-2">
            <LinkIcon className="w-3 h-3 text-primary-light" />
            <span className="text-xs text-primary-light">
              Derived from {model.parent_model_id.slice(0, 12)}...
            </span>
          </div>
        </div>
      )}

      {/* Monitoring Status (external models) */}
      {isExternal && model.monitoring_enabled && (
        <div className="mb-4 p-2 bg-blue-600/10 border border-blue-600/30 rounded">
          <div className="flex items-center gap-2">
            <Activity className="w-3 h-3 text-blue-400" />
            <span className="text-xs text-blue-400">
              Real-time monitoring active
            </span>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between pt-4 border-t border-gray-800">
        <div className="flex items-center gap-1 text-xs text-gray-500">
          <Calendar className="w-3 h-3" />
          <span>{formatDate(model.created_at)}</span>
        </div>
        
        {model.bias_status === 'critical' && (
          <div className="flex items-center gap-1 text-xs text-status-danger">
            <AlertCircle className="w-3 h-3" />
            <span>Needs attention</span>
          </div>
        )}
        
        {model.bias_status === 'compliant' && (
          <div className="flex items-center gap-1 text-xs text-status-success">
            <TrendingUp className="w-3 h-3" />
            <span>Compliant</span>
          </div>
        )}

        {/* Show mitigated status in footer */}
        {isMitigated && (
          <div className="flex items-center gap-1 text-xs text-primary-light">
            <Sparkles className="w-3 h-3" />
            <span>Bias mitigated</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelCard;