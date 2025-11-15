// frontend/src/components/chat/ChatInput.tsx
import React, { useState, useRef } from 'react';
import type { KeyboardEvent } from 'react';
import { Send, Loader2 } from 'lucide-react';
import type { ChatInputProps } from '../../types/index.ts';

const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  disabled = false,
  placeholder = 'Ask about compliance, models, or regulations...'
}) => {
  const [message, setMessage] = useState('');
  const isComposing = useRef(false);

  const handleSend = () => {
    const trimmed = message.trim();
    if (trimmed && !disabled) {
      onSendMessage(trimmed);
      setMessage('');
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (isComposing.current) return;
    
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t border-border bg-background-primary p-4">
      <div className="flex gap-3 items-end">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          onCompositionStart={() => { isComposing.current = true; }}
          onCompositionEnd={() => { isComposing.current = false; }}
          disabled={disabled}
          placeholder={placeholder}
          rows={1}
          className="flex-1 bg-slate-700 border border-slate-600 rounded-xl px-4 py-3 text-sm text-text-primary placeholder-text-muted focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent resize-none disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          style={{ minHeight: '48px', maxHeight: '120px' }}
          onInput={(e) => {
            const target = e.target as HTMLTextAreaElement;
            target.style.height = '48px';
            target.style.height = `${Math.min(target.scrollHeight, 120)}px`;
          }}
        />
        
        <button
          onClick={handleSend}
          disabled={disabled || !message.trim()}
          className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-primary to-primary-dark hover:from-primary-hover hover:to-primary text-white rounded-xl flex items-center justify-center transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-primary/20 hover:shadow-primary/40 disabled:shadow-none"
        >
          {disabled ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Send className="w-5 h-5" />
          )}
        </button>
      </div>

      <p className="text-xs text-text-muted mt-3 flex items-center gap-4">
        <span className="flex items-center gap-1.5">
          <kbd className="px-2 py-1 bg-slate-700 border border-slate-600 rounded text-xs font-medium">Enter</kbd>
          <span>to send</span>
        </span>
        <span className="flex items-center gap-1.5">
          <kbd className="px-2 py-1 bg-slate-700 border border-slate-600 rounded text-xs font-medium">Shift+Enter</kbd>
          <span>new line</span>
        </span>
      </p>
    </div>
  );
};

export default ChatInput;