// frontend/src/pages/ABTesting.tsx
import React, { useState, useEffect } from 'react';
import { Play, Pause, TrendingUp, Users, BarChart3, AlertCircle, CheckCircle} from 'lucide-react';
import type { ModelAB, ABTest} from '../types';

const ABTesting: React.FC = () => {
  const [models, setModels] = useState<ModelAB[]>([]);
  const [activeTests, setActiveTests] = useState<ABTest[]>([]);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [,setSelectedTest] = useState<ABTest | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  
  // Form state for new test
  const [newTest, setNewTest] = useState({
    model_a_id: '',
    model_b_id: '',
    test_name: '',
    traffic_split: 0.5,
    success_metric: 'accuracy',
    min_sample_size: 1000
  });

  useEffect(() => {
    fetchModels();
    fetchActiveTests();
    // Refresh active tests every 5 seconds
    const interval = setInterval(fetchActiveTests, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchModels = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/v1/models');
      const data = await response.json();
      setModels(data.models || []);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  };

  const fetchActiveTests = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/v1/ab-test/active');
      const data = await response.json();
      setActiveTests(data.tests || []);
    } catch (error) {
      console.error('Failed to fetch active tests:', error);
    }
  };

  const createABTest = async () => {
    setIsCreating(true);
    try {
    // Fix: Send as query parameters, not JSON body
    const params = new URLSearchParams({
      model_a_id: newTest.model_a_id,
      model_b_id: newTest.model_b_id,
      test_name: newTest.test_name,
      traffic_split: newTest.traffic_split.toString(),
      success_metric: newTest.success_metric,
      min_sample_size: newTest.min_sample_size.toString()
    });

    const response = await fetch(`http://localhost:8001/api/v1/ab-test/create?${params}`, {
      method: 'POST'
    });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Test created:', data);
        setShowCreateModal(false);
        fetchActiveTests();
        // Reset form
        setNewTest({
          model_a_id: '',
          model_b_id: '',
          test_name: '',
          traffic_split: 0.5,
          success_metric: 'accuracy',
          min_sample_size: 1000
        });
      }
    } catch (error) {
      console.error('Failed to create A/B test:', error);
    } finally {
      setIsCreating(false);
    }
  };

  const stopTest = async (testId: string) => {
    try {
      const response = await fetch(`http://localhost:8001/api/v1/ab-test/${testId}/stop`, {
        method: 'POST'
      });
      
      if (response.ok) {
        fetchActiveTests();
      }
    } catch (error) {
      console.error('Failed to stop test:', error);
    }
  };

  const fetchTestResults = async (testId: string) => {
    try {
      const response = await fetch(`http://localhost:8001/api/v1/ab-test/${testId}/results`);
      const data = await response.json();
      setSelectedTest(data.test);
    } catch (error) {
      console.error('Failed to fetch test results:', error);
    }
  };

  const getProgressPercentage = (test: ABTest): number => {
    const totalPredictions = test.results.model_a.predictions + test.results.model_b.predictions;
    return Math.min((totalPredictions / 2000) * 100, 100); // Assuming 2000 as target
  };

  const getWinnerColor = (model: 'model_a' | 'model_b', winner?: string): string => {
    if (!winner) return 'bg-gray-100';
    if (winner === model) return 'bg-green-100 border-green-500';
    return 'bg-red-50 border-red-300';
  };

  return (
    <div className="min-h-screen bg-background-primary p-6">
      {/* Header */}
      <div className="bg-card-background rounded-lg p-6 mb-6 border border-border">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-text-primary">A/B Testing Dashboard</h1>
            <p className="text-text-secondary mt-1">Compare model performance in production</p>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-hover flex items-center gap-2"
          >
            <Play className="w-4 h-4" />
            New A/B Test
          </button>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-card-background rounded-lg p-4 border border-border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-text-secondary">Active Tests</p>
              <p className="text-2xl font-bold text-text-primary">{activeTests.length}</p>
            </div>
            <BarChart3 className="w-8 h-8 text-primary" />
          </div>
        </div>
        
        <div className="bg-card-background rounded-lg p-4 border border-border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-text-secondary">Total Predictions</p>
              <p className="text-2xl font-bold text-text-primary">
                {activeTests.reduce((sum, test) => 
                  sum + test.results.model_a.predictions + test.results.model_b.predictions, 0
                )}
              </p>
            </div>
            <Users className="w-8 h-8 text-green-500" />
          </div>
        </div>

        <div className="bg-card-background rounded-lg p-4 border border-border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-text-secondary">Significant Results</p>
              <p className="text-2xl font-bold text-text-primary">
                {activeTests.filter(t => t.results.statistical_significance?.significant).length}
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-primary" />
          </div>
        </div>

        <div className="bg-card-background rounded-lg p-4 border border-border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-text-secondary">Avg Improvement</p>
              <p className="text-2xl font-bold text-text-primary">
                {activeTests.length > 0 ? 
                  `${((activeTests.reduce((sum, test) => {
                    const improvement = test.results.model_b.accuracy - test.results.model_a.accuracy;
                    return sum + improvement;
                  }, 0) / activeTests.length) * 100).toFixed(1)}%`
                  : '0%'
                }
              </p>
            </div>
            <CheckCircle className="w-8 h-8 text-orange-500" />
          </div>
        </div>
      </div>

      {/* Active Tests */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {activeTests.map((test) => (
          <div key={test.test_id} className="bg-card-background rounded-lg border border-border p-6">
            {/* Test Header */}
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-lg font-semibold text-text-primary">{test.test_name}</h3>
                <p className="text-sm text-text-secondary">Started: {new Date(test.created_at).toLocaleDateString()}</p>
              </div>
              <div className="flex gap-2">
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                  test.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                }`}>
                  {test.status.toUpperCase()}
                </span>
                {test.status === 'active' && (
                  <button
                    onClick={() => stopTest(test.test_id)}
                    className="p-1 hover:bg-red-50 rounded"
                    title="Stop Test"
                  >
                    <Pause className="w-4 h-4 text-red-500" />
                  </button>
                )}
              </div>
            </div>

            {/* Models Comparison */}
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className={`p-4 rounded-lg border-2 ${getWinnerColor('model_a', test.results.winner)}`}>
                <p className="text-xs text-text-secondary mb-1">Model A (Control)</p>
                <p className="font-medium text-text-primary">{test.model_a.model_type}</p>
                <div className="mt-2 space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Accuracy:</span>
                    <span className="font-medium">{(test.results.model_a.accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Samples:</span>
                    <span className="font-medium">{test.results.model_a.predictions}</span>
                  </div>
                </div>
              </div>

              <div className={`p-4 rounded-lg border-2 ${getWinnerColor('model_b', test.results.winner)}`}>
                <p className="text-xs text-text-secondary mb-1">Model B (Variant)</p>
                <p className="font-medium text-text-primary">{test.model_b.model_type}</p>
                <div className="mt-2 space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Accuracy:</span>
                    <span className="font-medium">{(test.results.model_b.accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-secondary">Samples:</span>
                    <span className="font-medium">{test.results.model_b.predictions}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Traffic Split */}
            <div className="mb-4">
              <p className="text-sm text-text-secondary mb-1">Traffic Split</p>
              <div className="flex h-6 bg-gray-200 rounded overflow-hidden">
                <div 
                  className="bg-blue-500 flex items-center justify-center text-white text-xs"
                  style={{ width: `${(1 - test.traffic_split) * 100}%` }}
                >
                  {((1 - test.traffic_split) * 100).toFixed(0)}%
                </div>
                <div 
                  className="bg-purple-500 flex items-center justify-center text-white text-xs"
                  style={{ width: `${test.traffic_split * 100}%` }}
                >
                  {(test.traffic_split * 100).toFixed(0)}%
                </div>
              </div>
            </div>

            {/* Progress Bar */}
            <div className="mb-4">
              <div className="flex justify-between text-sm mb-1">
                <span className="text-text-secondary">Test Progress</span>
                <span className="text-text-primary">{getProgressPercentage(test).toFixed(0)}%</span>
              </div>
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-primary transition-all duration-300"
                  style={{ width: `${getProgressPercentage(test)}%` }}
                />
              </div>
            </div>

            {/* Statistical Significance */}
            {test.results.statistical_significance && (
              <div className="p-3 bg-background-primary rounded-lg">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {test.results.statistical_significance.significant ? (
                      <CheckCircle className="w-5 h-5 text-green-500" />
                    ) : (
                      <AlertCircle className="w-5 h-5 text-yellow-500" />
                    )}
                    <span className="text-sm font-medium text-text-primary">
                      {test.results.statistical_significance.significant ? 'Statistically Significant' : 'Not Yet Significant'}
                    </span>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-text-secondary">p-value</p>
                    <p className="text-sm font-medium text-text-primary">
                      {test.results.statistical_significance.p_value.toFixed(4)}
                    </p>
                  </div>
                </div>
                
                {test.results.winner && (
                  <div className="mt-2 pt-2 border-t border-border">
                    <p className="text-sm text-text-primary">
                      <span className="font-medium">Winner: </span>
                      {test.results.winner === 'model_a' ? test.model_a.model_type : test.model_b.model_type}
                      {' '}
                      <span className="text-green-600">
                        (+{Math.abs(test.results.model_b.accuracy - test.results.model_a.accuracy).toFixed(2)}%)
                      </span>
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* View Details Button */}
            <button
              onClick={() => fetchTestResults(test.test_id)}
              className="w-full mt-4 px-4 py-2 bg-background-primary text-primary rounded hover:bg-gray-100 transition"
            >
              View Detailed Results
            </button>
          </div>
        ))}
      </div>

      {/* Empty State */}
      {activeTests.length === 0 && (
        <div className="bg-card-background rounded-lg border border-border p-12 text-center">
          <BarChart3 className="w-16 h-16 mx-auto text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-text-primary mb-2">No Active A/B Tests</h3>
          <p className="text-text-secondary mb-6">Create your first A/B test to compare model performance</p>
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-6 py-2 bg-primary text-white rounded-lg hover:bg-primary-hover"
          >
            Create A/B Test
          </button>
        </div>
      )}

      {/* Create Test Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-card-background rounded-lg p-6 max-w-md w-full">
            <h2 className="text-xl font-semibold text-text-primary mb-4">Create New A/B Test</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-text-primary mb-1">Test Name</label>
                <input
                  type="text"
                  value={newTest.test_name}
                  onChange={(e) => setNewTest({...newTest, test_name: e.target.value})}
                  className="w-full px-3 py-2 bg-background-primary border border-border rounded-lg text-text-primary"
                  placeholder="e.g., XGBoost vs Random Forest"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-text-primary mb-1">Model A (Control)</label>
                <select
                  value={newTest.model_a_id}
                  onChange={(e) => setNewTest({...newTest, model_a_id: e.target.value})}
                  className="w-full px-3 py-2 bg-background-primary border border-border rounded-lg text-text-primary"
                >
                  <option value="">Select Model A</option>
                  {models.map(model => (
                    <option key={model.model_id} value={model.model_id}>
                      {model.model_type} - {model.model_id}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-text-primary mb-1">Model B (Variant)</label>
                <select
                  value={newTest.model_b_id}
                  onChange={(e) => setNewTest({...newTest, model_b_id: e.target.value})}
                  className="w-full px-3 py-2 bg-background-primary border border-border rounded-lg text-text-primary"
                  disabled={!newTest.model_a_id}
                >
                  <option value="">Select Model B</option>
                  {models.filter(m => m.model_id !== newTest.model_a_id).map(model => (
                    <option key={model.model_id} value={model.model_id}>
                      {model.model_type} - {model.model_id}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-text-primary mb-1">
                  Traffic Split (% to Model B): {(newTest.traffic_split * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={newTest.traffic_split * 100}
                  onChange={(e) => setNewTest({...newTest, traffic_split: parseInt(e.target.value) / 100})}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-text-primary mb-1">Minimum Sample Size</label>
                <input
                  type="number"
                  value={newTest.min_sample_size}
                  onChange={(e) => setNewTest({...newTest, min_sample_size: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 bg-background-primary border border-border rounded-lg text-text-primary"
                />
              </div>
            </div>

            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowCreateModal(false)}
                className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
              >
                Cancel
              </button>
              <button
                onClick={createABTest}
                disabled={!newTest.test_name || !newTest.model_a_id || !newTest.model_b_id || isCreating}
                className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-hover disabled:opacity-50"
              >
                {isCreating ? 'Creating...' : 'Start Test'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ABTesting;