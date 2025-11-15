// frontend/src/contexts/AuthContext.tsx
import React, { createContext, useState, useEffect} from 'react';
import type { ReactNode } from 'react';
import { authApi } from '../services/api';

interface User {
  id: string;
  username: string;
  email: string;
  organization_id: string;
  organization_name?: string;
  is_superuser: boolean;
  is_active: boolean;
  role: 'admin' | 'member';
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, email: string, password: string, organizationName: string) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
}

export const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Load token from localStorage on mount
  useEffect(() => {
    const storedToken = localStorage.getItem('auth_token');
    if (storedToken) {
      setToken(storedToken);
      // Fetch user data
      fetchUser(storedToken);
    } else {
      setIsLoading(false);
    }
  }, []);

  // Fetch current user data
  const fetchUser = async (authToken: string) => {
    try {
      const userData = await authApi.getCurrentUser(authToken);
      setUser({
      ...userData,
      role: userData.role || 'member'  // ✅ Ensure role exists
      });
    } catch (error) {
      console.error('Failed to fetch user:', error);
      // Token might be invalid
      logout();
    } finally {
      setIsLoading(false);
    }
  };

  // Login
  const login = async (username: string, password: string) => {
    try {
      const response = await authApi.login(username, password);
      const { access_token, user: userData } = response;
      
      // Store token
      localStorage.setItem('auth_token', access_token);
      setToken(access_token);
      setUser({
      ...userData,
      role: userData.role || 'member'  // ✅ Ensure role exists
      });
    } catch (error: any) {
      console.error('Login failed:', error);
      throw new Error(error.response?.data?.detail || 'Login failed');
    }
  };

  // Register
  const register = async (
    username: string,
    email: string,
    password: string,
    organizationName: string
  ) => {
    try {
      const response = await authApi.register(username, email, password, organizationName);
      const { access_token, user: userData } = response;
      
      // Store token
      localStorage.setItem('auth_token', access_token);
      setToken(access_token);
      setUser({
      ...userData,
      role: userData.role || 'member'  // ✅ Ensure role exists
      });
    } catch (error: any) {
      console.error('Registration failed:', error);
      throw new Error(error.response?.data?.detail || 'Registration failed');
    }
  };

  // Logout
  const logout = () => {
    localStorage.removeItem('auth_token');
    setToken(null);
    setUser(null);
  };

  // Refresh user data
  const refreshUser = async () => {
    if (token) {
      await fetchUser(token);
    }
  };

  const value: AuthContextType = {
    user,
    token,
    isLoading,
    isAuthenticated: !!user && !!token,
    login,
    register,
    logout,
    refreshUser,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};