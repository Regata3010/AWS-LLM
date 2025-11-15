import axios, { AxiosError } from 'axios';
import type {
  Model,
  ModelsResponse,
  BiasAnalysis,
  BiasHistoryResponse,
  DashboardSummary,
  RecentActivityResponse,
  DatasetUploadResponse,
  ColumnSelectionRequest,
  ColumnSelectionResponse,
  ColumnOverrideRequest,
  ColumnOverrideResponse,
  TrainingRequest,
  TrainingResponse,
  BiasDetectionRequest,
  BiasDetectionResponse,
  MitigationRequest,
  MitigationResponse,
  MitigationInfoResponse,
  DetectTaskTypeRequest,
  DetectTaskTypeResponse,
  RegisterExternalModelRegisterRequest,
  RegisterExternalModelRegisterResponse,
  PredictionLogRequest,
  PredictionLogResponse,
  BatchPredictionLogRequest,
  BatchPredictionLogResponse,
  AnalyzeRequest,
  AnalyzeResponse,
  ExternalModel,
  ReportGenerateRequest,
  ReportGenerateResponse,
} from '../types';

// ============================================
// AXIOS INSTANCE CONFIGURATION
// ============================================

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60 seconds (training can be slow)
});

// ============================================
// REQUEST/RESPONSE INTERCEPTORS
// ============================================

// Request interceptor (for adding auth tokens later)
apiClient.interceptors.request.use(
  (config) => {
    // Could add auth token here:
    // const token = localStorage.getItem('auth_token');
    // if (token) config.headers.Authorization = `Bearer ${token}`;
    
    console.log(`→ ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor (for error handling)
apiClient.interceptors.response.use(
  (response) => {
    console.log(`← ${response.config.method?.toUpperCase()} ${response.config.url} - ${response.status}`);
    return response;
  },
  (error: AxiosError) => {
    // Handle common errors
    if (error.response) {
      console.error(`API Error ${error.response.status}:`, error.response.data);
      
      // Specific error handling
      if (error.response.status === 404) {
        console.error('Resource not found');
      } else if (error.response.status === 500) {
        console.error('Server error:', (error.response.data as any)?.detail);
      }
    } else if (error.request) {
      console.error('No response from server - is backend running?');
    } else {
      console.error('Request setup error:', error.message);
    }
    
    return Promise.reject(error);
  }
);

// ============================================
// MODEL MANAGEMENT APIs
// ============================================

export const modelApi = {
  /**
   * List all models with optional filtering
   * GET /api/v1/models
   */
  list: async (params?: {
    skip?: number;
    limit?: number;
    bias_status?: string;
  }): Promise<ModelsResponse> => {
    const response = await apiClient.get('/models', { params });
    return response.data;
  },

  /**
   * Get specific model by ID
   * GET /api/v1/model/{id}
   */
  getById: async (id: string): Promise<Model> => {
    const response = await apiClient.get(`/model/${id}`);
    return response.data;
  },

  getMitigationInfo: async (modelId: string): Promise<MitigationInfoResponse | null> => {
    try {
      const response = await apiClient.get<MitigationInfoResponse>(`/model/${modelId}/mitigation`);
      return response.data;
    } catch (error: any) {
      // Return null if no mitigation found (404 is expected)
      if (error.response?.status === 404) {
        return null;
      }
      throw error;
    }
  },


  /**
   * Delete a model
   * DELETE /api/v1/model/{id}
   */
  delete: async (id: string): Promise<{ status: string; message: string }> => {
    const response = await apiClient.delete(`/model/${id}`);
    return response.data;
  },
};

// ============================================
// AUTH APIs
// ============================================

export const authApi = {
  /**
   * Login user
   * POST /api/v1/auth/login
   */
  login: async (username: string, password: string): Promise<{
    access_token: string;
    token_type: string;
    user: {
      id: string;
      username: string;
      email: string;
      organization_id: string;
      organization_name?: string;
      is_superuser: boolean;
      is_active: boolean;
      role: 'admin' | 'member';
    };
  }> => {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    
    const response = await apiClient.post('/login', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
    return response.data;
  },

  /**
   * Register new user
   * POST /api/v1/auth/register
   */
  register: async (
    username: string,
    email: string,
    password: string,
    organizationName: string
  ): Promise<{
    access_token: string;
    token_type: string;
    user: {
      id: string;
      username: string;
      email: string;
      organization_id: string;
      organization_name?: string;
      is_superuser: boolean;
      is_active: boolean;
      role: 'admin' | 'member';
    };
  }> => {
    const response = await apiClient.post('/register', {
      username,
      email,
      password,
      organization_name: organizationName,
    });
    return response.data;
  },

  /**
   * Get current user
   * GET /api/v1/auth/me
   */
  getCurrentUser: async (token: string): Promise<{
    id: string;
    username: string;
    email: string;
    organization_id: string;
    organization_name?: string;
    is_superuser: boolean;
    is_active: boolean;
    role: 'admin' | 'member';
  }> => {
    const response = await apiClient.get('/me', {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    return response.data;
  },

  /**
   * Logout (client-side only, clear token)
   */
  logout: (): void => {
    localStorage.removeItem('auth_token');
  },
};

// ============================================
// REQUEST INTERCEPTOR - AUTO-ATTACH JWT
// ============================================

// Update your existing request interceptor:
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token to all requests
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    console.log(`→ ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Update response interceptor to handle 401 Unauthorized:
apiClient.interceptors.response.use(
  (response) => {
    console.log(`← ${response.config.method?.toUpperCase()} ${response.config.url} - ${response.status}`);
    return response;
  },
  (error: AxiosError) => {
    // Handle 401 - redirect to login
    if (error.response?.status === 401) {
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    
    // Rest of your existing error handling...
    if (error.response) {
      console.error(`API Error ${error.response.status}:`, error.response.data);
    } else if (error.request) {
      console.error('No response from server - is backend running?');
    } else {
      console.error('Request setup error:', error.message);
    }
    
    return Promise.reject(error);
  }
);








// ============================================
// BIAS DETECTION APIs
// ============================================

export const biasApi = {
  /**
   * Run bias detection on a model
   * POST /api/v1/bias
   */
  detect: async (request: BiasDetectionRequest): Promise<BiasDetectionResponse> => {
    const response = await apiClient.post('/bias', request);
    return response.data;
  },

  /**
   * Get latest bias analysis for a model
   * GET /api/v1/bias/latest/{model_id}
   */
  getLatest: async (modelId: string): Promise<BiasAnalysis> => {
    const response = await apiClient.get(`/bias/latest/${modelId}`);
    return response.data;
  },

  /**
   * Get bias analysis history for a model
   * GET /api/v1/bias/history/{model_id}
   */
  getHistory: async (modelId: string): Promise<BiasHistoryResponse> => {
    const response = await apiClient.get(`/bias/history/${modelId}`);
    return response.data;
  },

  /**
   * Apply bias mitigation
   * POST /api/v1/bias/mitigate
   */
  mitigate: async (request: MitigationRequest): Promise<MitigationResponse> => {
    const response = await apiClient.post('/bias/mitigate', request);
    return response.data;
  },
};

// ============================================
// DASHBOARD APIs
// ============================================

export const dashboardApi = {
  /**
   * Get dashboard summary statistics
   * GET /api/v1/dashboard/summary
   */
  getSummary: async (): Promise<DashboardSummary> => {
    const response = await apiClient.get('/dashboard/summary');
    return response.data;
  },

  /**
   * Get recent activity
   * GET /api/v1/dashboard/recent
   */
  getRecent: async (limit: number = 10): Promise<RecentActivityResponse> => {
    const response = await apiClient.get('/dashboard/recent', { 
      params: { limit } 
    });
    return response.data;
  },
};

// ============================================
// TRAINING & UPLOAD APIs
// ============================================

export const trainingApi = {
  /**
   * Upload CSV dataset
   * POST /api/v1/upload
   */
  upload: async (file: File): Promise<DatasetUploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await apiClient.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  /**
   * Select columns (with optional LLM)
   * POST /api/v1/analysis/columns/select
   */
  selectColumns: async (request: ColumnSelectionRequest): Promise<ColumnSelectionResponse> => {
    const response = await apiClient.post('columns/select', request);
    return response.data;
  },

  /**
   * Override column selections
   * POST /api/v1/analysis/columns/override
   */
  overrideColumns: async (request: ColumnOverrideRequest): Promise<ColumnOverrideResponse> => {
    const response = await apiClient.post('columns/override', request);
    return response.data;
  },

  /**
   * Detect task type for target column
   * POST /api/v1/training/detect-task-type
   */
  detectTaskType: async (request: DetectTaskTypeRequest): Promise<DetectTaskTypeResponse> => {
    const response = await apiClient.post('/training/detect-task-type', request);
    return response.data;
  },

  /**
   * Train a model
   * POST /api/v1/training
   */
  train: async (request: TrainingRequest): Promise<TrainingResponse> => {
    const response = await apiClient.post('/training', request);
    return response.data;
  },
};


// ============================================
// MONITORING APIs (v2.0)
// ============================================

export const monitoringApi = {
  /**
   * Register an external model for monitoring
   * POST /api/v1/models/register
   */
  registerExternalModel: async (
    request: RegisterExternalModelRegisterRequest
  ): Promise<RegisterExternalModelRegisterResponse> => {
    const response = await apiClient.post('/models/register', request);
    return response.data;
  },

  /**
   * Log a single prediction
   * POST /api/v1/monitor/log
   */
  logPrediction: async (
    request: PredictionLogRequest
  ): Promise<PredictionLogResponse> => {
    const response = await apiClient.post('/monitor/log', request);
    return response.data;
  },

  /**
   * Log batch predictions
   * POST /api/v1/monitor/batch
   */
  logBatchPredictions: async (
    request: BatchPredictionLogRequest
  ): Promise<BatchPredictionLogResponse> => {
    const response = await apiClient.post('/monitor/batch', request);
    return response.data;
  },

  /**
   * Upload prediction logs via CSV
   * POST /api/v1/monitor/upload_csv
   */
  uploadPredictionCSV: async (
    modelId: string,
    file: File
  ): Promise<any> => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await apiClient.post(
      `/monitor/upload_csv?model_id=${modelId}`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  },

  /**
   * Analyze bias in prediction logs
   * POST /api/v1/analyze
   */
  analyzeBias: async (
    request: AnalyzeRequest
  ): Promise<AnalyzeResponse> => {
    const response = await apiClient.post('/analyze', request);
    return response.data;
  },

  /**
   * Get monitoring statistics for a model
   * GET /api/v1/monitor/stats/{model_id}
   */
  getModelStats: async (modelId: string): Promise<any> => {
    const response = await apiClient.get(`/monitor/stats/${modelId}`);
    return response.data;
  },

  /**
   * List all models (both external and trained)
   * GET /api/v1/models
   */
  listAllModels: async (params?: {
    skip?: number;
    limit?: number;
    source?: 'external' | 'trained';
    status?: string;
  }): Promise<any> => {
    const response = await apiClient.get('/models', { params });
    return response.data;
  },

  /**
   * Get external model by ID
   * GET /api/v1/models/{model_id}
   */
  getExternalModel: async (modelId: string): Promise<ExternalModel> => {
    const response = await apiClient.get(`/models/${modelId}`);
    return response.data;
  },

  /**
   * Update external model
   * PUT /api/v1/models/{model_id}
   */
  updateExternalModel: async (
    modelId: string,
    updates: Partial<RegisterExternalModelRegisterRequest>
  ): Promise<any> => {
    const response = await apiClient.put(`/models/${modelId}`, updates);
    return response.data;
  },

  /**
   * Delete external model
   * DELETE /api/v1/models/{model_id}
   */
  deleteExternalModel: async (modelId: string): Promise<any> => {
    const response = await apiClient.delete(`/models/${modelId}`);
    return response.data;
  },

  /**
   * Generate compliance report
   * POST /api/v1/reports/generate/{model_id}
   */
  generateReport: async (
    modelId: string,
    request: ReportGenerateRequest
  ): Promise<ReportGenerateResponse> => {
    const response = await apiClient.post(
      `/reports/generate/${modelId}`,
      request
    );
    return response.data;
  },

  /**
   * Download report
   * GET /api/v1/reports/download/{report_id}
   */
  downloadReport: async (reportId: string): Promise<Blob> => {
    const response = await apiClient.get(`/reports/download/${reportId}`, {
      responseType: 'blob',
    });
    return response.data;
  },
};

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Check if backend is reachable
 */
export const checkBackendHealth = async (): Promise<boolean> => {
  try {
    const response = await axios.get('http://localhost:8001/health', {
      timeout: 5000,
    });
    return response.status === 200;
  } catch {
    return false;
  }
};

/**
 * Get MLflow URL for a run
 */
export const getMLflowUrl = (runId: string, experimentId: string = '1'): string => {
  return `http://localhost:5000/#/experiments/${experimentId}/runs/${runId}`;
};

// ============================================
// EXPORT DEFAULT CLIENT
// ============================================

export default apiClient;