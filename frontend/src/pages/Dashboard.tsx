import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  Activity,

} from 'lucide-react';
import { dashboardApi, modelApi } from '../services/api';
import Card from '../components/ui/Card';
import HealthCard from '../components/features/HealthCard';
import ModelCard from '../components/features/ModelCard';
import DonutChart from '../components/charts/DonutChart';

const Dashboard: React.FC = () => {
  // Fetch dashboard summary
  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: ['dashboard-summary'],
    queryFn: dashboardApi.getSummary,
  });

  // Fetch models
  const { data: modelsData, isLoading: modelsLoading } = useQuery({
    queryKey: ['models'],
    queryFn: () => modelApi.list({ limit: 100 }),
  });

  if (summaryLoading || modelsLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
          <p className="text-gray-400">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  // Calculate task distribution for donut chart
  const taskDistribution = modelsData?.models.reduce((acc, model) => {
    const task = model.task_type || 'unknown';
    acc[task] = (acc[task] || 0) + 1;
    return acc;
  }, {} as Record<string, number>) || {};

  const donutData = [
    { name: 'Binary', value: taskDistribution['binary'] || 0, color: '#6366F1' },
    { name: 'Multiclass', value: taskDistribution['multiclass'] || 0, color: '#8B5CF6' },
    { name: 'Continuous', value: taskDistribution['continuous'] || 0, color: '#EC4899' },
  ].filter(item => item.value > 0);

  // Get models by status for display
  const criticalModels = modelsData?.models.filter(m => m.bias_status === 'critical') || [];
  const compliantModels = modelsData?.models.filter(m => m.bias_status === 'compliant') || [];

  return (
    <div className="min-h-screen bg-background-primary">
      {/* Page Header */}
      <div className="px-8 py-6 border-b border-gray-800">
        <h1 className="text-3xl font-bold text-white">Dashboard</h1>
        <p className="text-gray-400 mt-1">Monitor bias detection across all models</p>
      </div>

      <div className="p-8">
        {/* Health Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <HealthCard
            title="Total Models"
            value={summary?.total_models || 0}
            subtitle={`${modelsData?.models.length || 0} showing`}
            icon={Shield}
            variant="default"
          />
          
          <HealthCard
            title="Compliant Models"
            value={summary?.compliant_models || 0}
            subtitle={`${summary?.compliance_rate || 0}% compliance rate`}
            icon={CheckCircle}
            variant="success"
          />
          
          <HealthCard
            title="Models at Risk"
            value={summary?.models_at_risk || 0}
            subtitle="Warning level bias"
            icon={Activity}
            variant="warning"
          />
          
          <HealthCard
            title="Critical Issues"
            value={summary?.critical_models || 0}
            subtitle="Requires immediate action"
            icon={AlertTriangle}
            variant="danger"
          />
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Task Distribution */}
          <Card 
            title="Task Distribution" 
            subtitle={`${modelsData?.total || 0} total models`}
            className="lg:col-span-1"
          >
            <DonutChart
              data={donutData}
              centerLabel="Models"
              centerValue={modelsData?.total || 0}
            />
          </Card>

          {/* Compliance Overview */}
          <Card 
            title="Compliance Overview" 
            subtitle="Model status breakdown"
            className="lg:col-span-2"
          >
            <div className="space-y-4">
              {/* Compliant */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-400">Compliant</span>
                  <span className="text-sm font-semibold text-status-success">
                    {summary?.compliant_models || 0} models
                  </span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-2">
                  <div
                    className="bg-status-success h-2 rounded-full transition-all duration-500"
                    style={{
                      width: `${summary?.total_models ? (summary.compliant_models / summary.total_models) * 100 : 0}%`,
                    }}
                  />
                </div>
              </div>

              {/* Warning */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-400">Warning</span>
                  <span className="text-sm font-semibold text-status-warning">
                    {summary?.models_at_risk || 0} models
                  </span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-2">
                  <div
                    className="bg-status-warning h-2 rounded-full transition-all duration-500"
                    style={{
                      width: `${summary?.total_models ? (summary.models_at_risk / summary.total_models) * 100 : 0}%`,
                    }}
                  />
                </div>
              </div>

              {/* Critical */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-400">Critical</span>
                  <span className="text-sm font-semibold text-status-danger">
                    {summary?.critical_models || 0} models
                  </span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-2">
                  <div
                    className="bg-status-danger h-2 rounded-full transition-all duration-500"
                    style={{
                      width: `${summary?.total_models ? (summary.critical_models / summary.total_models) * 100 : 0}%`,
                    }}
                  />
                </div>
              </div>

              {/* Unknown */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-400">Not Analyzed</span>
                  <span className="text-sm font-semibold text-gray-400">
                    {summary?.unknown_status || 0} models
                  </span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-2">
                  <div
                    className="bg-gray-600 h-2 rounded-full transition-all duration-500"
                    style={{
                      width: `${summary?.total_models ? (summary.unknown_status / summary.total_models) * 100 : 0}%`,
                    }}
                  />
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* Critical Models Section */}
        {criticalModels.length > 0 && (
          <Card 
            title="Critical Models" 
            subtitle={`${criticalModels.length} models require immediate attention`}
            className="mb-8"
            actions={
              <span className="text-sm text-gray-400">
                {criticalModels.length} of {modelsData?.total}
              </span>
            }
          >
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {criticalModels.slice(0, 6).map((model) => (
                <ModelCard
                  key={model.model_id}
                  model={model}
                />
              ))}
            </div>
            {criticalModels.length > 6 && (
              <div className="mt-4 text-center">
                <button className="text-indigo-400 hover:text-indigo-300 text-sm font-medium">
                  View all {criticalModels.length} critical models →
                </button>
              </div>
            )}
          </Card>
        )}

        {/* Compliant Models Section */}
        {compliantModels.length > 0 && (
          <Card 
            title="Compliant Models" 
            subtitle={`${compliantModels.length} models meeting bias requirements`}
            className="mb-8"
            actions={
              <span className="text-sm text-gray-400">
                {compliantModels.length} of {modelsData?.total}
              </span>
            }
          >
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {compliantModels.slice(0, 6).map((model) => (
                <ModelCard
                  key={model.model_id}
                  model={model}
                />
              ))}
            </div>
            {compliantModels.length > 6 && (
              <div className="mt-4 text-center">
                <button className="text-green-400 hover:text-green-300 text-sm font-medium">
                  View all {compliantModels.length} compliant models →
                </button>
              </div>
            )}
          </Card>
        )}

        {/* All Models Section */}
        <Card 
          title="All Models" 
          subtitle={`${modelsData?.total || 0} models in system`}
          actions={
            <div className="flex gap-2">
              <button className="px-3 py-1 text-sm bg-gray-800 text-gray-300 rounded hover:bg-gray-700 transition-colors">
                Filter
              </button>
              <button className="px-3 py-1 text-sm bg-gray-800 text-gray-300 rounded hover:bg-gray-700 transition-colors">
                Sort
              </button>
            </div>
          }
        >
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {modelsData?.models.slice(0, 12).map((model) => (
              <ModelCard
                key={model.model_id}
                model={model}
              />
            ))}
          </div>
          
          {(modelsData?.total || 0) > 12 && (
            <div className="mt-6 text-center">
              <button className="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 transition-colors font-medium">
                Load More Models
              </button>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;