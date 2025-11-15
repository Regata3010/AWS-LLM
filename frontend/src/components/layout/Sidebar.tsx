// frontend/src/components/layout/Sidebar.tsx
import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  Home, 
  Upload, 
  BarChart3, 
  Settings, 
  Shield, 
  Activity, 
  FileText, 
  Users, 
  LogOut, 
  User,
  Crown,
  UserPlus
} from 'lucide-react';
import { useAuth } from '../../hooks/useAuth';

interface MenuItem {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  path: string;
  adminOnly?: boolean;  // Only show to admins
}

const Sidebar: React.FC = () => {
  const { user, logout } = useAuth();

  const menuItems: MenuItem[] = [
    { icon: Home, label: 'Dashboard', path: '/' },
    { icon: Upload, label: 'Monitor Models', path: '/upload' },
    { icon: BarChart3, label: 'Models', path: '/models' },
    { icon: Activity, label: 'Live Monitor', path: '/monitor' },
    { icon: Users, label: 'A/B Testing', path: '/ab-testing' },
    { icon: FileText, label: 'Reports', path: '/reports' },
    { icon: UserPlus, label: 'Invite Users', path: '/invite', adminOnly: true },  // Admin only
    { icon: Settings, label: 'Settings', path: '/settings', adminOnly: true },    // Admin only
  ];

  const handleLogout = () => {
    logout();
    window.location.href = '/login';
  };

  // Filter menu items based on role
  const visibleMenuItems = menuItems.filter(item => {
    if (!item.adminOnly) return true;  // Show to everyone
    return user?.role === 'admin' || user?.is_superuser;  // Show only to admins
  });

  return (
    <div className="w-64 h-screen bg-background-secondary border-r border-gray-800 flex flex-col">
      {/* Logo Section */}
      <div className="p-6 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-indigo-600 rounded-lg flex items-center justify-center">
            <Shield className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">BiasGuard</h1>
            <p className="text-xs text-gray-400">AI Fairness Platform</p>
          </div>
        </div>
      </div>

      {/* User Info */}
      {user && (
        <div className="p-4 border-b border-gray-800">
          <div className="flex items-center gap-3 p-3 bg-background-card rounded-lg">
            <div className="w-10 h-10 bg-indigo-600/20 rounded-full flex items-center justify-center">
              <User className="w-5 h-5 text-indigo-400" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-white truncate">{user.username}</p>
              <div className="flex items-center gap-2 mt-1">
                {user.is_superuser ? (
                  <span className="inline-flex items-center gap-1 text-xs text-yellow-400">
                    <Crown className="w-3 h-3" />
                    Superuser
                  </span>
                ) : (
                  <span className={`inline-flex items-center gap-1 text-xs ${
                    user.role === 'admin' ? 'text-indigo-400' : 'text-gray-400'
                  }`}>
                    {user.role === 'admin' && <Shield className="w-3 h-3" />}
                    {user.role === 'admin' ? 'Admin' : 'Member'}
                  </span>
                )}
              </div>
              <p className="text-xs text-gray-500 truncate mt-0.5">
                {user.organization_name || 'Organization'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Navigation Menu */}
      <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
        {visibleMenuItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                  isActive
                    ? 'bg-indigo-600 text-white shadow-lg'
                    : 'text-gray-400 hover:bg-background-card hover:text-white'
                }`
              }
            >
              <Icon className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
            </NavLink>
          );
        })}
      </nav>

      {/* Logout Button */}
      <div className="p-4 border-t border-gray-800">
        <button
          onClick={handleLogout}
          className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-gray-400 hover:bg-red-900/20 hover:text-red-400 transition-all duration-200"
        >
          <LogOut className="w-5 h-5" />
          <span className="font-medium">Logout</span>
        </button>
      </div>

      {/* Status Footer */}
      <div className="p-4 border-t border-gray-800">
        <div className="bg-background-card rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-gray-400">Backend Status</span>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 bg-status-success rounded-full animate-pulse"></div>
              <span className="text-xs text-status-success">Connected</span>
            </div>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-400">WebSocket</span>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 bg-status-success rounded-full animate-pulse"></div>
              <span className="text-xs text-status-success">Live</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;