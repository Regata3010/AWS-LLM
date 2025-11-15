import React from 'react';
import { Target, Shield, Sparkles } from 'lucide-react';
import type { ColumnAnalysisResponse } from '../../types';

interface ColumnSelectorProps {
  columns: string[];
  targetColumn: string;
  sensitiveColumns: string[];
  onTargetChange: (column: string) => void;
  onSensitiveChange: (columns: string[]) => void;
  aiSuggested?: Pick<ColumnAnalysisResponse, 'suggested_target' | 'suggested_sensitive'>;
}

const ColumnSelector: React.FC<ColumnSelectorProps> = ({
  columns,
  targetColumn,
  sensitiveColumns,
  onTargetChange,
  onSensitiveChange,
  aiSuggested,
}) => {
  const toggleSensitiveColumn = (column: string) => {
    if (sensitiveColumns.includes(column)) {
      onSensitiveChange(sensitiveColumns.filter((c) => c !== column));
    } else {
      onSensitiveChange([...sensitiveColumns, column]);
    }
  };

  const isAISuggested = (column: string, type: 'target' | 'sensitive'): boolean => {
    if (!aiSuggested) return false;
    if (type === 'target') return aiSuggested.suggested_target === column || false;
    return aiSuggested.suggested_sensitive?.includes(column) || false;  // ✅ Safe with ?.
  };

  return (
    <div className="space-y-6">
      {/* Target Column Selection */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <Target className="w-5 h-5 text-indigo-400" />
          <h3 className="text-lg font-semibold text-white">Target Column</h3>
          <span className="text-xs text-gray-500">(What you're predicting)</span>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {columns.map((column) => {
            const isSelected = targetColumn === column;
            const isAI = isAISuggested(column, 'target');
            
            return (
              <button
                key={column}
                onClick={() => onTargetChange(column)}
                disabled={sensitiveColumns.includes(column)}
                className={`relative p-3 rounded-lg border transition-all text-left ${
                  isSelected
                    ? 'bg-indigo-600 border-indigo-500 text-white'
                    : sensitiveColumns.includes(column)
                    ? 'bg-gray-800 border-gray-700 text-gray-500 cursor-not-allowed'
                    : 'bg-background-card border-gray-700 text-gray-300 hover:border-gray-600'
                }`}
              >
                {isAI && !isSelected && (
                  <div className="absolute -top-1 -right-1">
                    <div className="bg-indigo-500 rounded-full p-1">
                      <Sparkles className="w-3 h-3 text-white" />
                    </div>
                  </div>
                )}
                
                <div className="font-medium truncate">{column}</div>
                {isAI && !isSelected && (
                  <div className="text-xs text-indigo-400 mt-1">AI suggested</div>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Sensitive Columns Selection */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <Shield className="w-5 h-5 text-yellow-400" />
          <h3 className="text-lg font-semibold text-white">Sensitive Attributes</h3>
          <span className="text-xs text-gray-500">(Protected characteristics)</span>
        </div>
        
        <div className="bg-background-secondary rounded-lg p-4 mb-4">
          <p className="text-sm text-gray-400">
            Select attributes that should be protected from discrimination (e.g., age, race, gender).
            These will be used for fairness analysis.
          </p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {columns.map((column) => {
            const isSelected = sensitiveColumns.includes(column);
            const isTarget = targetColumn === column;
            const isAI = isAISuggested(column, 'sensitive');
            
            return (
              <button
                key={column}
                onClick={() => toggleSensitiveColumn(column)}
                disabled={isTarget}
                className={`relative p-3 rounded-lg border transition-all text-left ${
                  isSelected
                    ? 'bg-yellow-600 border-yellow-500 text-white'
                    : isTarget
                    ? 'bg-gray-800 border-gray-700 text-gray-500 cursor-not-allowed'
                    : 'bg-background-card border-gray-700 text-gray-300 hover:border-gray-600'
                }`}
              >
                {isAI && !isSelected && (
                  <div className="absolute -top-1 -right-1">
                    <div className="bg-yellow-500 rounded-full p-1">
                      <Sparkles className="w-3 h-3 text-white" />
                    </div>
                  </div>
                )}
                
                <div className="font-medium truncate">{column}</div>
                {isAI && !isSelected && (
                  <div className="text-xs text-yellow-400 mt-1">AI suggested</div>
                )}
                {isTarget && (
                  <div className="text-xs text-gray-500 mt-1">Target column</div>
                )}
              </button>
            );
          })}
        </div>

        {sensitiveColumns.length > 0 && (
          <div className="mt-4 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
            <p className="text-sm text-yellow-400">
              Selected {sensitiveColumns.length} sensitive attribute{sensitiveColumns.length > 1 ? 's' : ''}:
              <span className="font-semibold ml-2">{sensitiveColumns.join(', ')}</span>
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ColumnSelector;