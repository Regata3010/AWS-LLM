import React from 'react';
import { Cpu, Zap, TrendingUp } from 'lucide-react';

// interface ModelOption {
//   name: string;
//   description: string;
//   icon: any;
//   recommended?: boolean;
// }

interface ModelSelectorProps {
  taskType: string;
  selectedModel: string;
  onModelChange: (model: string) => void;
  recommendedModels: string[];
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  taskType,
  selectedModel,
  onModelChange,
  recommendedModels,
}) => {
  const modelDescriptions: Record<string, { description: string; icon: any }> = {
    'XGBoost': {
      description: 'Gradient boosting - High accuracy, handles imbalanced data well',
      icon: Zap,
    },
    'Random Forest': {
      description: 'Ensemble method - Robust, interpretable, handles missing values',
      icon: TrendingUp,
    },
    'Logistic Regression': {
      description: 'Linear model - Fast, interpretable, good for binary classification',
      icon: Cpu,
    },
    'Decision Tree': {
      description: 'Tree-based - Highly interpretable, good for feature analysis',
      icon: TrendingUp,
    },
    'SVM': {
      description: 'Support Vector Machine - Effective for high-dimensional data',
      icon: Cpu,
    },
    'XGBoost Regressor': {
      description: 'Gradient boosting for regression - High accuracy predictions',
      icon: Zap,
    },
    'Random Forest Regressor': {
      description: 'Ensemble regression - Robust continuous predictions',
      icon: TrendingUp,
    },
    'Linear Regression': {
      description: 'Linear regression - Simple, fast, interpretable',
      icon: Cpu,
    },
    'LightGBM Regressor': {
      description: 'Fast gradient boosting - Efficient for large datasets',
      icon: Zap,
    },
  };

  return (
    <div>
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-white mb-2">Select Model Algorithm</h3>
        <p className="text-sm text-gray-400">
          Task type detected: <span className="text-indigo-400 font-medium">{taskType}</span>
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {recommendedModels.map((modelName) => {
          const modelInfo = modelDescriptions[modelName] || {
            description: 'Machine learning model',
            icon: Cpu,
          };
          const Icon = modelInfo.icon;
          const isSelected = selectedModel === modelName;
          const isRecommended = recommendedModels[0] === modelName;

          return (
            <button
              key={modelName}
              onClick={() => onModelChange(modelName)}
              className={`relative p-5 rounded-lg border text-left transition-all ${
                isSelected
                  ? 'bg-indigo-600 border-indigo-500 shadow-lg'
                  : 'bg-background-card border-gray-700 hover:border-gray-600'
              }`}
            >
              {isRecommended && (
                <div className="absolute -top-2 -right-2">
                  <span className="bg-status-success text-white text-xs px-2 py-1 rounded-full font-medium">
                    Recommended
                  </span>
                </div>
              )}

              <div className="flex items-start gap-4">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
                  isSelected ? 'bg-white/20' : 'bg-gray-800'
                }`}>
                  <Icon className={`w-6 h-6 ${isSelected ? 'text-white' : 'text-gray-400'}`} />
                </div>

                <div className="flex-1 min-w-0">
                  <h4 className={`font-semibold mb-1 ${isSelected ? 'text-white' : 'text-gray-200'}`}>
                    {modelName}
                  </h4>
                  <p className={`text-sm ${isSelected ? 'text-indigo-100' : 'text-gray-400'}`}>
                    {modelInfo.description}
                  </p>
                </div>

                {/* Selection indicator */}
                <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center flex-shrink-0 ${
                  isSelected 
                    ? 'border-white bg-white' 
                    : 'border-gray-600'
                }`}>
                  {isSelected && (
                    <div className="w-2 h-2 rounded-full bg-indigo-600" />
                  )}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default ModelSelector;