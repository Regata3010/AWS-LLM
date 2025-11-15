// frontend/src/components/chat/MessageBubble.tsx
import React from 'react';
import { User, Bot } from 'lucide-react';
import ToolUsageIndicator from './ToolUsageIndicator';
import type { ChatMessage } from '../../services/chatApi';

interface MessageBubbleProps {
  message: ChatMessage;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`flex gap-4 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      {/* Avatar */}
      <div className={`flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center ${
        isUser 
          ? 'bg-gradient-to-br from-primary to-primary-dark shadow-lg shadow-primary/20' 
          : 'bg-slate-700 border border-slate-600'
      }`}>
        {isUser ? (
          <User className="w-5 h-5 text-white" />
        ) : (
          <Bot className="w-5 h-5 text-primary" />
        )}
      </div>

      {/* Message Content */}
      <div className={`flex-1 ${isUser ? 'items-end' : 'items-start'} flex flex-col`}>
        <div
          className={`max-w-[85%] rounded-2xl px-5 py-3.5 ${
            isUser
              ? 'bg-gradient-to-br from-primary to-primary-dark text-white shadow-lg shadow-primary/20'
              : 'bg-slate-700 text-text-primary border border-slate-600'
          }`}
        >
          <p className="text-sm whitespace-pre-wrap break-words leading-relaxed">
            {message.content}
          </p>
          
          {/* Tool Usage Indicators (AI messages only) */}
          {!isUser && message.tools_used && message.tools_used.length > 0 && (
            <ToolUsageIndicator tools={message.tools_used} />
          )}
        </div>

        {/* Timestamp */}
        <span className={`text-xs text-text-muted mt-2 px-2 ${isUser ? 'text-right' : 'text-left'}`}>
          {new Date(message.timestamp).toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
          })}
        </span>
      </div>
    </div>
  );
};

export default MessageBubble;