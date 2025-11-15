import React from 'react';
import { AlertCircle, CheckCircle, AlertTriangle, Info } from 'lucide-react';

interface MetricDisplayProps {
  name: string;
  value: number | string;
  threshold?: number | number[];
  status: 'compliant' | 'warning' | 'critical' | 'info';
  interpretation: string;
  details?: {
    label: string;
    value: string | number;
  }[];
  showGauge?: boolean;
  gaugeMin?: number;
  gaugeMax?: number;
}

const MetricDisplay: React.FC<MetricDisplayProps> = ({
  name,
  value,
  threshold,
  status,
  interpretation,
  details,
  showGauge = false,
  gaugeMin = 0,
  gaugeMax = 1,
}) => {
  const statusConfig = {
    compliant: {
      color: 'status-success',
      icon: CheckCircle,
      bg: 'bg-status-success/10',
      border: 'border-status-success/30',
    },
    warning: {
      color: 'status-warning',
      icon: AlertTriangle,
      bg: 'bg-status-warning/10',
      border: 'border-status-warning/30',
    },
    critical: {
      color: 'status-danger',
      icon: AlertCircle,
      bg: 'bg-status-danger/10',
      border: 'border-status-danger/30',
    },
    info: {
      color: 'status-info',
      icon: Info,
      bg: 'bg-status-info/10',
      border: 'border-status-info/30',
    },
  };

  const config = statusConfig[status] || statusConfig.info;  // ✅ Fallback to 'info'
  const Icon = config?.icon || Info;

  // Calculate gauge position (for disparate impact 0-2 range)
  const getGaugePosition = () => {
    if (typeof value !== 'number') return 50;
    const normalizedValue = ((value - gaugeMin) / (gaugeMax - gaugeMin)) * 100;
    return Math.max(0, Math.min(100, normalizedValue));
  };

  return (
    <div className={`bg-background-card border ${config.border} rounded-lg p-5 ${config.bg}`}>
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h4 className="text-white font-semibold text-lg mb-1">{name}</h4>
          <p className="text-sm text-gray-400">{interpretation}</p>
        </div>
        <div className={`flex items-center gap-2 px-3 py-1 rounded-full bg-${config.color}/20`}>
          <Icon className={`w-4 h-4 text-${config.color}`} />
          <span className={`text-sm font-medium text-${config.color} capitalize`}>
            {status}
          </span>
        </div>
      </div>

      {/* Value Display */}
      <div className="mb-4">
        <div className="flex items-baseline gap-2">
          <span className="text-4xl font-bold text-white">
            {typeof value === 'number' ? value.toFixed(3) : value}
          </span>
          {threshold && (
            <span className="text-sm text-gray-400">
              / threshold: {Array.isArray(threshold) ? `${threshold[0]}-${threshold[1]}` : threshold}
            </span>
          )}
        </div>
      </div>

      {/* Gauge (for Disparate Impact) */}
      {showGauge && typeof value === 'number' && (
        <div className="mb-4">
          <div className="relative h-3 bg-gray-800 rounded-full overflow-hidden">
            {/* Threshold zones */}
            <div className="absolute inset-0 flex">
              <div className="flex-1 bg-status-danger/20" style={{ width: '40%' }}></div>
              <div className="flex-1 bg-status-success/20" style={{ width: '45%' }}></div>
              <div className="flex-1 bg-status-danger/20" style={{ width: '15%' }}></div>
            </div>
            
            {/* Value indicator */}
            <div
              className="absolute top-0 bottom-0 w-1 bg-white shadow-lg"
              style={{ left: `${getGaugePosition()}%` }}
            />
          </div>
          
          {/* Labels */}
          <div className="flex justify-between mt-2 text-xs text-gray-500">
            <span>{gaugeMin}</span>
            <span>0.8 (min)</span>
            <span>1.0</span>
            <span>1.25 (max)</span>
            <span>{gaugeMax}</span>
          </div>
        </div>
      )}

      {/* Details */}
      {details && details.length > 0 && (
        <div className="grid grid-cols-2 gap-3 mt-4 pt-4 border-t border-gray-800">
          {details.map((detail, index) => (
            <div key={index}>
              <p className="text-xs text-gray-500 mb-1">{detail.label}</p>
              <p className="text-sm font-semibold text-white">
                {typeof detail.value === 'number' 
                  ? detail.value.toFixed(4) 
                  : detail.value}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default MetricDisplay;