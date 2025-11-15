// frontend/src/components/chat/ToolUsageIndicator.tsx
import React from 'react';
import { Database, FileText, Search } from 'lucide-react';
import type { ToolUsageIndicatorProps } from '../../types/index.ts';

const ToolUsageIndicator: React.FC<ToolUsageIndicatorProps> = ({ tools }) => {
  if (!tools || tools.length === 0) return null;

  const getToolInfo = (tool: string) => {
    const lower = tool.toLowerCase();
    
    if (lower.includes('rag') || lower.includes('regulation')) {
      return {
        icon: <FileText className="w-3 h-3" />,
        label: 'RAG',
        color: 'bg-primary/10 text-primary-light border-primary/30'
      };
    }
    
    if (lower.includes('cag') || lower.includes('model') || lower.includes('database')) {
      return {
        icon: <Database className="w-3 h-3" />,
        label: 'CAG',
        color: 'bg-blue-500/10 text-blue-400 border-blue-500/30'
      };
    }
    
    if (lower.includes('search')) {
      return {
        icon: <Search className="w-3 h-3" />,
        label: 'Search',
        color: 'bg-green-500/10 text-green-400 border-green-500/30'
      };
    }
    
    return {
      icon: <Database className="w-3 h-3" />,
      label: tool,
      color: 'bg-gray-500/10 text-gray-400 border-gray-500/30'
    };
  };

  return (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {tools.map((tool, idx) => {
        const info = getToolInfo(tool);
        return (
          <div
            key={idx}
            className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border ${info.color}`}
          >
            {info.icon}
            <span>{info.label}</span>
          </div>
        );
      })}
    </div>
  );
};

export default ToolUsageIndicator;