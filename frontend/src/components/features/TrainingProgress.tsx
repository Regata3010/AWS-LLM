import React from 'react';
import { CheckCircle, Loader, AlertCircle } from 'lucide-react';

type TrainingStage = 'preprocessing' | 'training' | 'evaluating' | 'complete' | 'error';

interface TrainingProgressProps {
  stage: TrainingStage;
  progress: number;
  message?: string;
  error?: string;
}

const TrainingProgress: React.FC<TrainingProgressProps> = ({
  stage,
  progress,
  message,
  error,
}) => {
  const stages = [
    { id: 'preprocessing', label: 'Preprocessing Data', icon: Loader },
    { id: 'training', label: 'Training Model', icon: Loader },
    { id: 'evaluating', label: 'Evaluating Performance', icon: Loader },
    { id: 'complete', label: 'Complete', icon: CheckCircle },
  ];

  const getCurrentStageIndex = () => {
    return stages.findIndex((s) => s.id === stage);
  };

  const currentIndex = getCurrentStageIndex();

  return (
    <div className="bg-background-card rounded-lg border border-gray-800 p-6">
      {error ? (
        // Error State
        <div className="text-center py-8">
          <div className="w-16 h-16 bg-status-danger/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <AlertCircle className="w-8 h-8 text-status-danger" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">Training Failed</h3>
          <p className="text-sm text-gray-400 max-w-md mx-auto">{error}</p>
        </div>
      ) : (
        <>
          {/* Progress Bar */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">
                {stage === 'complete' ? 'Training Complete' : message || 'Training in progress...'}
              </span>
              <span className="text-sm font-semibold text-indigo-400">{progress}%</span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-2">
              <div
                className="bg-indigo-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>

          {/* Stage Indicators */}
          <div className="space-y-3">
            {stages.map((stageItem, index) => {
              const Icon = stageItem.icon;
              const isActive = index === currentIndex;
              const isComplete = index < currentIndex || stage === 'complete';

              return (
                <div
                  key={stageItem.id}
                  className={`flex items-center gap-4 p-3 rounded-lg transition-all ${
                    isActive ? 'bg-indigo-600/20' : ''
                  }`}
                >
                  {/* Icon */}
                  <div
                    className={`w-10 h-10 rounded-full flex items-center justify-center ${
                      isComplete
                        ? 'bg-status-success/20 text-status-success'
                        : isActive
                        ? 'bg-indigo-600/20 text-indigo-400'
                        : 'bg-gray-800 text-gray-600'
                    }`}
                  >
                    {isComplete ? (
                      <CheckCircle className="w-5 h-5" />
                    ) : isActive ? (
                      <Icon className="w-5 h-5 animate-spin" />
                    ) : (
                      <Icon className="w-5 h-5" />
                    )}
                  </div>

                  {/* Label */}
                  <div className="flex-1">
                    <p
                      className={`font-medium ${
                        isComplete || isActive ? 'text-white' : 'text-gray-500'
                      }`}
                    >
                      {stageItem.label}
                    </p>
                    {isActive && message && (
                      <p className="text-xs text-gray-400 mt-1">{message}</p>
                    )}
                  </div>

                  {/* Status */}
                  {isComplete && (
                    <span className="text-xs text-status-success font-medium">✓ Done</span>
                  )}
                  {isActive && (
                    <span className="text-xs text-indigo-400 font-medium">In progress...</span>
                  )}
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
};

export default TrainingProgress;