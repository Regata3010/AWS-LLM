import React from 'react';
import { type LucideIcon } from 'lucide-react';

interface HealthCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  trend?: {
    value: number;
    positive: boolean;
  };
  variant?: 'default' | 'success' | 'warning' | 'danger';
}

const HealthCard: React.FC<HealthCardProps> = ({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  variant = 'default',
}) => {
  const variantStyles = {
    default: 'border-gray-800',
    success: 'border-status-success/30 bg-status-success/5',
    warning: 'border-status-warning/30 bg-status-warning/5',
    danger: 'border-status-danger/30 bg-status-danger/5',
  };

  const iconColors = {
    default: 'bg-gray-700 text-gray-400',
    success: 'bg-status-success/20 text-status-success',
    warning: 'bg-status-warning/20 text-status-warning',
    danger: 'bg-status-danger/20 text-status-danger',
  };

  return (
    <div className={`bg-background-card rounded-lg border p-6 ${variantStyles[variant]} hover:border-gray-700 transition-all`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm text-gray-400 mb-1">{title}</p>
          <div className="flex items-baseline gap-2">
            <h2 className="text-3xl font-bold text-white">{value}</h2>
            {trend && (
              <span className={`text-sm font-medium ${trend.positive ? 'text-status-success' : 'text-status-danger'}`}>
                {trend.positive ? '↑' : '↓'} {Math.abs(trend.value)}%
              </span>
            )}
          </div>
          {subtitle && (
            <p className="text-xs text-gray-500 mt-1">{subtitle}</p>
          )}
        </div>
        
        <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${iconColors[variant]}`}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
    </div>
  );
};

export default HealthCard;