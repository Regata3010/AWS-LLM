// frontend/src/components/chat/ChatWidget.tsx
import React, { useState } from 'react';
import { MessageCircle, X } from 'lucide-react';
import ChatPanel from './ChatPanel';
import type { ChatWidgetProps } from '../../types/index.ts';


const ChatWidget: React.FC<ChatWidgetProps> = ({ modelId }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [hasUnread, setHasUnread] = useState(false);

  const toggleChat = () => {
    setIsOpen(!isOpen);
    if (!isOpen) {
      setHasUnread(false);
    }
  };

  return (
    <>
      {/* Floating Button - Redesigned */}
      <button
        onClick={toggleChat}
        className={`fixed bottom-6 right-6 w-16 h-16 rounded-full shadow-xl transition-all duration-300 flex items-center justify-center z-[9999] group ${
          isOpen 
            ? 'bg-slate-700 hover:bg-slate-600' 
            : 'bg-gradient-to-br from-primary to-primary-dark hover:from-primary-hover hover:to-primary shadow-primary/20 hover:shadow-primary/40'
        }`}
        aria-label="Toggle chat"
      >
        {isOpen ? (
          <X className="w-6 h-6 text-white transition-transform group-hover:rotate-90" />
        ) : (
          <>
            <MessageCircle className="w-6 h-6 text-white transition-transform group-hover:scale-110" />
            {hasUnread && (
              <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full border-2 border-background-primary flex items-center justify-center">
                <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
              </span>
            )}
          </>
        )}
      </button>

      {/* Chat Panel - NO OVERLAY */}
      <ChatPanel 
        isOpen={isOpen} 
        onClose={() => setIsOpen(false)} 
        modelId={modelId}
      />
    </>
  );
};

export default ChatWidget;