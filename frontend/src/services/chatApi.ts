// frontend/src/lib/chatApi.ts
import apiClient from './api';
import type { ChatRequest, ChatResponse } from '../types/index.ts';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  tools_used?: string[];
}

export const chatApi = {
  /**
   * Send a message to the AI agent
   * POST /api/v1/agent/chat
   */
  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const response = await apiClient.post('/chat', request);
    return response.data;
  },

  /**
   * Get chat history (coming soon)
   * GET /api/v1/agent/chat/history
   */
  getHistory: async (threadId?: string): Promise<ChatMessage[]> => {
    const response = await apiClient.get('/chat/history', {
      params: { thread_id: threadId }
    });
    return response.data.messages || [];
  },

  /**
   * Clear chat history
   * DELETE /api/v1/agent/chat/history
   */
  clearHistory: async (threadId?: string): Promise<void> => {
    await apiClient.delete('/chat/history', {
      params: { thread_id: threadId }
    });
  },

  /**
   * Check agent health
   * GET /api/v1/agent/health
   */
  healthCheck: async (): Promise<{
    status: string;
    checks: Record<string, boolean>;
    message: string;
  }> => {
    const response = await apiClient.get('/health');
    return response.data;
  }
};