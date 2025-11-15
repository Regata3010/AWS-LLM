// frontend/src/App.tsx
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import Sidebar from './components/layout/Sidebar';
import Dashboard from './pages/Dashboard';
import ModelDetail from './pages/ModelDetail';
import Models from './pages/Models';
import Monitor from './pages/Monitor';
import Reports from './pages/Reports';
import Settings from './pages/Settings';
import RealTimeMonitor from './pages/RealTimeMonitor';
import ABTesting from './pages/ABTesting';
import Login from './pages/Login';
import Register from './pages/Register';
import InviteUser from './pages/InviteUser';
import ChatWidget from './components/chat/ChatWidget';


function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public Routes */}
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />

          {/* Protected Routes */}
          <Route
            path="/*"
            element={
              <ProtectedRoute>
                <div className="flex h-screen bg-background-primary overflow-hidden">
                  <Sidebar />
                  <div className="flex-1 overflow-auto">
                    <Routes>
                      <Route path="/" element={<Dashboard />} />
                      <Route path="/models" element={<Models />} />
                      <Route path="/model/:id" element={<ModelDetail />} />
                      <Route path="/upload" element={<Monitor />} />
                      <Route path="/monitor" element={<RealTimeMonitor />} />
                      <Route path="/ab-testing" element={<ABTesting />} />
                      <Route path="/reports" element={<Reports />} />
                      <Route path="/settings" element={<Settings />} />
                      <Route path="/invite" element={<InviteUser />} />
                      <Route path="*" element={<Navigate to="/" replace />} />
                    </Routes>
                  </div>
                  <div className="fixed bottom-20 right-6 bg-red-500 text-white p-4 rounded">
                    Chat with AI Agent
                  </div>
                  <ChatWidget />

                </div>
              </ProtectedRoute>
            }
          />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;