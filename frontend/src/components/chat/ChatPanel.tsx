// frontend/src/components/chat/ChatPanel.tsx
import React, { useState, useRef, useEffect } from 'react';
import { X, Trash2, AlertCircle, CheckCircle, Sparkles, RefreshCw } from 'lucide-react';
import MessageBubble from './MessageBubble';
import ChatInput from './ChatInput';
import { chatApi } from '../../services/chatApi';
import type { ChatMessage } from '../../services/chatApi';
import type { ChatPanelProps } from '../../types/index.ts';



const ChatPanel: React.FC<ChatPanelProps> = ({ isOpen, onClose, modelId }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [agentHealth, setAgentHealth] = useState<'healthy' | 'degraded' | 'unhealthy'>('healthy');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const threadId = useRef<string>(crypto.randomUUID());

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (isOpen) {
      checkAgentHealth();
    }
  }, []);

  const checkAgentHealth = async () => {
    try {
      const health = await chatApi.healthCheck();
      setAgentHealth(health.status as 'healthy' | 'degraded' | 'unhealthy');
    } catch (err) {
      setAgentHealth('unhealthy');
    }
  };

  const handleSendMessage = async (content: string) => {
    setError(null);

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await chatApi.sendMessage({
        message: content,
        model_id: modelId,
        thread_id: threadId.current
      });

      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: response.response,
        timestamp: response.timestamp,
        tools_used: response.tools_used
      };
      setMessages(prev => [...prev, assistantMessage]);

    } catch (err: any) {
      console.error('Chat error:', err);
      setError(err.response?.data?.detail || 'Failed to get response from agent');
      
      const errorMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearHistory = async () => {
    if (window.confirm('Clear all conversation history?')) {
      try {
        await chatApi.clearHistory(threadId.current);
        setMessages([]);
        threadId.current = crypto.randomUUID();
      } catch (err) {
        console.error('Failed to clear history:', err);
      }
    }
  };

  const getHealthIndicator = () => {
    switch (agentHealth) {
      case 'healthy':
        return { icon: <CheckCircle className="w-4 h-4" />, color: 'text-green-400', bg: 'bg-green-500/10' };
      case 'degraded':
        return { icon: <AlertCircle className="w-4 h-4" />, color: 'text-yellow-400', bg: 'bg-yellow-500/10' };
      case 'unhealthy':
        return { icon: <AlertCircle className="w-4 h-4" />, color: 'text-red-400', bg: 'bg-red-500/10' };
    }
  };

  const healthStatus = getHealthIndicator();

  return (
    <div
      className={`fixed top-0 right-0 h-full w-full md:w-[480px] bg-background-primary border-l border-border shadow-2xl z-[9999] transform transition-transform duration-300 ease-in-out ${
        isOpen ? 'translate-x-0' : 'translate-x-full'
      }`}
    >
      {/* Header - Redesigned */}
      <div className="bg-gradient-to-r from-primary to-primary-dark px-6 py-5 border-b border-primary-dark/50">
        <div className="flex justify-between items-start">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 bg-white/10 backdrop-blur-sm rounded-xl flex items-center justify-center border border-white/20">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">BiasGuard AI</h2>
              <div className="flex items-center gap-2 mt-1">
                <div className={`flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium ${healthStatus.bg} ${healthStatus.color}`}>
                  {healthStatus.icon}
                  <span className="capitalize">{agentHealth}</span>
                </div>
                <span className="text-xs text-white/60">Compliance Assistant</span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-1">
            <button
              onClick={checkAgentHealth}
              className="p-2 hover:bg-white/10 rounded-lg transition"
              title="Refresh status"
            >
              <RefreshCw className="w-4 h-4 text-white/80" />
            </button>
            <button
              onClick={handleClearHistory}
              className="p-2 hover:bg-white/10 rounded-lg transition"
              title="Clear history"
            >
              <Trash2 className="w-4 h-4 text-white/80" />
            </button>
            <button
              onClick={onClose}
              className="p-2 hover:bg-white/10 rounded-lg transition"
            >
              <X className="w-5 h-5 text-white" />
            </button>
          </div>
        </div>

        {modelId && (
          <div className="mt-3 px-3 py-2 bg-white/5 backdrop-blur-sm rounded-lg border border-white/10">
            <p className="text-xs text-white/70">
              <span className="font-medium text-white">Context:</span> 
              <span className="ml-2 font-mono">{modelId.slice(0, 20)}...</span>
            </p>
          </div>
        )}
      </div>

      {/* Messages Area - Solid Background */}
      <div 
        className="flex-1 overflow-y-auto p-6 space-y-6 bg-background-secondary" 
        style={{ height: 'calc(100vh - 240px)' }}
      >
        {messages.length === 0 ? (
          <div className="text-center py-16">
            <div className="w-20 h-20 bg-gradient-to-br from-primary to-primary-dark rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg shadow-primary/20">
              <Sparkles className="w-10 h-10 text-white" />
            </div>
            <h3 className="text-xl font-bold text-text-primary mb-3">
              Ask me anything!
            </h3>
            <p className="text-sm text-text-secondary max-w-xs mx-auto mb-8 leading-relaxed">
              I can help with compliance questions, model analysis, and regulatory guidance.
            </p>
            
            <div className="space-y-3 max-w-sm mx-auto">
              <p className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-4">
                Try asking:
              </p>
              {[
                'What models do I have?',
                'Is my loan model compliant?',
                "What's the four-fifths rule?",
                'Explain ECOA Section 1002.6(a)'
              ].map((example, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSendMessage(example)}
                  className="w-full text-left px-5 py-3.5 bg-background-primary hover:bg-slate-700 border border-border hover:border-primary rounded-xl transition-all text-sm text-text-primary font-medium group"
                >
                  <span className="text-primary group-hover:text-primary-light mr-2">→</span>
                  {example}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            
            {isLoading && (
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-slate-700 flex items-center justify-center border border-slate-600">
                  <Sparkles className="w-5 h-5 text-primary" />
                </div>
                <div className="bg-slate-700 border border-slate-600 rounded-2xl px-5 py-4">
                  <div className="flex gap-1.5">
                    <div className="w-2.5 h-2.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-2.5 h-2.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-2.5 h-2.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </>
        )}

        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-red-400 mb-1">Error</p>
                <p className="text-sm text-red-300">{error}</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Area - Solid Background */}
      <ChatInput
        onSendMessage={handleSendMessage}
        disabled={isLoading || agentHealth === 'unhealthy'}
        placeholder={
          modelId 
            ? `Ask about model ${modelId.slice(0, 8)}...`
            : 'Ask about compliance, models, or regulations...'
        }
      />
    </div>
  );
};

export default ChatPanel;