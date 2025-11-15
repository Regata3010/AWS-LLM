import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  User, 
  Shield, 
  Building2, 
  Mail, 
  Lock, 
  Crown,
  AlertCircle,
  UserPlus,
  Calendar
} from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import Card from '../components/ui/Card';

const Settings: React.FC = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [showPasswordModal, setShowPasswordModal] = useState(false);

  if (!user) {
    return null;
  }

  const isAdmin = user.role === 'admin' || user.is_superuser;

  return (
    <div className="min-h-screen bg-background-primary">
      {/* Header */}
      <div className="px-8 py-6 border-b border-gray-800">
        <h1 className="text-3xl font-bold text-white">Settings</h1>
        <p className="text-gray-400 mt-1">Manage your account and organization settings</p>
      </div>

      <div className="p-8 max-w-4xl">
        {/* User Profile Section */}
        <Card title="User Profile">
          <div className="space-y-6">
            {/* Username */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Username
              </label>
              <div className="flex items-center gap-3 px-4 py-3 bg-gray-900/50 border border-gray-700 rounded-lg">
                <User className="w-5 h-5 text-gray-500" />
                <span className="text-white">{user.username}</span>
              </div>
            </div>

            {/* Email */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Email
              </label>
              <div className="flex items-center gap-3 px-4 py-3 bg-gray-900/50 border border-gray-700 rounded-lg">
                <Mail className="w-5 h-5 text-gray-500" />
                <span className="text-white">{user.email}</span>
              </div>
            </div>

            {/* Role */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Role
              </label>
              <div className="flex items-center gap-3 px-4 py-3 bg-gray-900/50 border border-gray-700 rounded-lg">
                {user.is_superuser ? (
                  <>
                    <Crown className="w-5 h-5 text-yellow-400" />
                    <span className="text-yellow-400 font-semibold">Superuser</span>
                  </>
                ) : (
                  <>
                    <Shield className="w-5 h-5 text-indigo-400" />
                    <span className="text-white capitalize">{user.role}</span>
                  </>
                )}
              </div>
            </div>

            {/* Organization */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Organization
              </label>
              <div className="flex items-center gap-3 px-4 py-3 bg-gray-900/50 border border-gray-700 rounded-lg">
                <Building2 className="w-5 h-5 text-gray-500" />
                <span className="text-white">{user.organization_name || 'N/A'}</span>
              </div>
            </div>

            {/* Change Password */}
            <div className="pt-4 border-t border-gray-700">
              <button
                onClick={() => setShowPasswordModal(true)}
                className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 flex items-center gap-2"
              >
                <Lock className="w-4 h-4" />
                Change Password
              </button>
            </div>
          </div>
        </Card>

        {/* Organization Settings (Admin Only) */}
        {isAdmin && (
          <div className="mt-6">
            <Card title="Organization Settings">
              <div className="space-y-6">
                {/* Org Name */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Organization Name
                  </label>
                  <div className="flex items-center gap-3 px-4 py-3 bg-gray-900/50 border border-gray-700 rounded-lg">
                    <Building2 className="w-5 h-5 text-gray-500" />
                    <span className="text-white">{user.organization_name}</span>
                  </div>
                </div>

                {/* Plan Info */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Plan
                  </label>
                  <div className="p-4 bg-gray-900/50 border border-gray-700 rounded-lg">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-white font-medium">Free Plan</span>
                      <span className="px-2 py-1 bg-indigo-600/20 text-indigo-400 rounded text-xs">
                        Active
                      </span>
                    </div>
                    <div className="text-sm text-gray-400 space-y-1">
                      <p>• Up to 50 users</p>
                      <p>• Up to 1,000 models</p>
                      <p>• Unlimited predictions</p>
                    </div>
                  </div>
                </div>

                {/* Team Management */}
                <div className="pt-4 border-t border-gray-700">
                  <div className="flex justify-between items-center mb-4">
                    <h4 className="text-white font-medium">Team Members</h4>
                    <button
                      onClick={() => navigate('/invite')}
                      className="px-3 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 flex items-center gap-2 text-sm"
                    >
                      <UserPlus className="w-4 h-4" />
                      Invite User
                    </button>
                  </div>
                  
                  <div className="p-4 bg-blue-900/20 border border-blue-600/30 rounded-lg">
                    <div className="flex items-start gap-2">
                      <AlertCircle className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                      <p className="text-sm text-blue-300">
                        Team member management coming soon. Use the Invite Users page to add team members.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Created Date */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Organization Created
                  </label>
                  <div className="flex items-center gap-3 px-4 py-3 bg-gray-900/50 border border-gray-700 rounded-lg">
                    <Calendar className="w-5 h-5 text-gray-500" />
                    <span className="text-white">
                      {new Date().toLocaleDateString()}
                    </span>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        )}

        {/* Member Notice */}
        {!isAdmin && (
          <div className="mt-6 p-4 bg-blue-900/20 border border-blue-600/30 rounded-lg">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm text-blue-300 font-medium mb-1">
                  Member Account
                </p>
                <p className="text-xs text-gray-400">
                  You're a member of <strong>{user.organization_name}</strong>. 
                  Contact your organization admin to manage settings, invite users, or upgrade your plan.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Password Change Modal (Placeholder) */}
      {showPasswordModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full border border-gray-700">
            <h3 className="text-xl font-bold text-white mb-4">Change Password</h3>
            <div className="p-4 bg-blue-900/20 border border-blue-600/30 rounded-lg mb-4">
              <p className="text-sm text-blue-300">
                Password change functionality coming soon.
              </p>
            </div>
            <button
              onClick={() => setShowPasswordModal(false)}
              className="w-full px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Settings;