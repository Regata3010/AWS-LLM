// frontend/src/pages/InviteUser.tsx
import React, { useState } from 'react';
import { useAuth } from '../hooks/useAuth';
import { UserPlus, Mail, Shield, User as UserIcon, AlertCircle, CheckCircle, Copy } from 'lucide-react';
import Card from '../components/ui/Card';

const InviteUser: React.FC = () => {
  const { user } = useAuth();
  const [email, setEmail] = useState('');
  const [role, setRole] = useState<'admin' | 'member'>('member');
  const [inviteToken, setInviteToken] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  // Check if user is admin
  if (!user?.is_superuser && user?.role !== 'admin') {
    return (
      <div className="min-h-screen bg-background-primary flex items-center justify-center p-4">
        <Card title="Access Denied">
          <div className="text-center py-8">
            <AlertCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
            <h2 className="text-xl font-bold text-white mb-2">Admin Access Required</h2>
            <p className="text-gray-400">Only administrators can invite users to the organization.</p>
          </div>
        </Card>
      </div>
    );
  }

  const handleInvite = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess(false);
    setIsLoading(true);

    try {
      // TODO: Replace with actual API call when backend endpoint is ready
      // const response = await authApi.inviteUser(email, role);
      
      // For now, generate a fake token
      const fakeToken = `invite_${Math.random().toString(36).substr(2, 9)}`;
      setInviteToken(fakeToken);
      setSuccess(true);
      setEmail('');
      
      // Placeholder - will implement real API call later
      console.log('Inviting user:', { email, role, organization_id: user.organization_id });
    } catch (err: any) {
      setError(err.message || 'Failed to send invitation');
    } finally {
      setIsLoading(false);
    }
  };

  const copyInviteLink = () => {
    const inviteUrl = `${window.location.origin}/register?token=${inviteToken}`;
    navigator.clipboard.writeText(inviteUrl);
    alert('Invite link copied to clipboard!');
  };

  return (
    <div className="min-h-screen bg-background-primary p-8">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Invite Users</h1>
          <p className="text-gray-400">
            Invite team members to join <span className="text-indigo-400 font-medium">{user.organization_name}</span>
          </p>
        </div>

        {/* Invitation Form */}
        <Card title="Send Invitation">
          <form onSubmit={handleInvite} className="space-y-6">
            {/* Error Message */}
            {error && (
              <div className="p-4 bg-red-900/20 border border-red-600/50 rounded-lg flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-red-400">{error}</p>
              </div>
            )}

            {/* Success Message */}
            {success && inviteToken && (
              <div className="p-4 bg-green-900/20 border border-green-600/50 rounded-lg">
                <div className="flex items-start gap-3 mb-3">
                  <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm text-green-400 font-medium mb-2">Invitation Created!</p>
                    <p className="text-xs text-gray-400 mb-3">
                      Copy this link and send it to the user (email functionality coming soon)
                    </p>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        readOnly
                        value={`${window.location.origin}/register?token=${inviteToken}`}
                        className="flex-1 px-3 py-2 bg-gray-900/50 border border-gray-700 rounded text-xs text-white font-mono"
                      />
                      <button
                        type="button"
                        onClick={copyInviteLink}
                        className="px-3 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-500 flex items-center gap-2"
                      >
                        <Copy className="w-4 h-4" />
                        Copy
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Email Field */}
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">
                Email Address *
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500" />
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  disabled={isLoading}
                  className="w-full pl-11 pr-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder:text-gray-500 focus:ring-2 focus:ring-indigo-500 disabled:opacity-50"
                  placeholder="colleague@company.com"
                />
              </div>
            </div>

            {/* Role Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-3">
                Role *
              </label>
              <div className="space-y-3">
                {/* Admin Option */}
                <label className="flex items-start gap-3 p-4 bg-gray-800 border-2 border-gray-700 rounded-lg cursor-pointer hover:border-indigo-600 transition">
                  <input
                    type="radio"
                    name="role"
                    value="admin"
                    checked={role === 'admin'}
                    onChange={(e) => setRole(e.target.value as 'admin')}
                    className="mt-1"
                  />
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <Shield className="w-4 h-4 text-indigo-400" />
                      <span className="font-medium text-white">Admin</span>
                    </div>
                    <p className="text-xs text-gray-400">
                      Full access: manage models, invite users, view all reports, configure settings
                    </p>
                  </div>
                </label>

                {/* Member Option */}
                <label className="flex items-start gap-3 p-4 bg-gray-800 border-2 border-gray-700 rounded-lg cursor-pointer hover:border-indigo-600 transition">
                  <input
                    type="radio"
                    name="role"
                    value="member"
                    checked={role === 'member'}
                    onChange={(e) => setRole(e.target.value as 'member')}
                    className="mt-1"
                  />
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <UserIcon className="w-4 h-4 text-gray-400" />
                      <span className="font-medium text-white">Member</span>
                    </div>
                    <p className="text-xs text-gray-400">
                      Standard access: create models, upload data, run analysis, view reports
                    </p>
                  </div>
                </label>
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 disabled:opacity-50 flex items-center justify-center gap-2 font-medium"
            >
              {isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Sending...
                </>
              ) : (
                <>
                  <UserPlus className="w-5 h-5" />
                  Send Invitation
                </>
              )}
            </button>
          </form>
        </Card>

        {/* Info Box */}
        <div className="mt-6 p-4 bg-blue-900/20 border border-blue-600/30 rounded-lg">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-gray-400">
              <p className="font-medium text-blue-300 mb-1">Note: Email functionality coming soon</p>
              <p>For now, manually share the invitation link with your team members. They'll be able to register with a pre-assigned role in your organization.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InviteUser;