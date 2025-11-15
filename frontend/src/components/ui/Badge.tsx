import React from 'react';

type BadgeVariant = 'success' | 'warning' | 'danger' | 'info' | 'default';

interface BadgeProps {
  children: React.ReactNode;
  variant: BadgeVariant;
  size?: 'sm' | 'md' | 'lg';
}

const Badge: React.FC<BadgeProps> = ({ children, variant, size = 'md' }) => {
  const variantStyles = {
    success: 'bg-status-success/20 text-status-success border-status-success/30',
    warning: 'bg-status-warning/20 text-status-warning border-status-warning/30',
    danger: 'bg-status-danger/20 text-status-danger border-status-danger/30',
    info: 'bg-status-info/20 text-status-info border-status-info/30',
    default: 'bg-gray-700/50 text-gray-300 border-gray-600',
  };

  const sizeStyles = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-3 py-1 text-sm',
    lg: 'px-4 py-1.5 text-base',
  };

  return (
    <span
      className={`inline-flex items-center rounded-full border font-medium ${variantStyles[variant]} ${sizeStyles[size]}`}
    >
      {children}
    </span>
  );
};

// Helper function to get badge variant from bias status
export const getBiasStatusVariant = (status: string): BadgeVariant => {
  switch (status) {
    case 'compliant':
      return 'success';
    case 'warning':
      return 'warning';
    case 'critical':
      return 'danger';
    case 'unknown':
      return 'default';
    default:
      return 'default';
  }
};

export default Badge;