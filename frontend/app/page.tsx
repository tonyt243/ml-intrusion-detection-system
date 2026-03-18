'use client';

import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

interface Detection {
  timestamp: string;
  is_attack: boolean;
  alert_level: string;
  reason: string;
  predictions: {
    random_forest?: {
      is_attack: boolean;
      confidence: number;
      attack_probability: number;
    };
    isolation_forest?: {
      is_attack: boolean;
      anomaly_score: number;
    };
  };
}

interface Statistics {
  total_packets: number;
  attacks_detected: number;
  normal_packets: number;
  attack_rate: number;
}

export default function Dashboard() {
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [recentDetections, setRecentDetections] = useState<Detection[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Fetch statistics
  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_URL}/statistics`);
      setStatistics(response.data);
      setIsConnected(true);
    } catch (error) {
      console.error('Error fetching stats:', error);
      setIsConnected(false);
    }
  };

  // Fetch recent detections
  const fetchDetections = async () => {
    try {
      const response = await axios.get(`${API_URL}/recent?limit=10`);
      setRecentDetections(response.data.recent_detections || []);
    } catch (error) {
      console.error('Error fetching detections:', error);
    }
  };

  // Manual refresh both
  const handleRefresh = async () => {
    setIsRefreshing(true);
    await Promise.all([fetchStats(), fetchDetections()]);
    setIsRefreshing(false);
  };

  // Initial load only
  useEffect(() => {
    fetchStats();
    fetchDetections();
  }, []);

  // Prepare data for pie chart
  const pieData = statistics ? [
    { name: 'Normal', value: statistics.normal_packets, color: '#10b981' },
    { name: 'Attacks', value: statistics.attacks_detected, color: '#ef4444' }
  ] : [];

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold mb-2">ML-Based Intrusion Detection System</h1>
              <p className="text-gray-400">Real-time network traffic analysis using machine learning</p>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
                <span className="text-sm text-gray-400">{isConnected ? 'Connected' : 'Disconnected'}</span>
              </div>
              <button
                onClick={handleRefresh}
                disabled={isRefreshing}
                className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                  isRefreshing
                    ? 'bg-gray-700 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700'
                }`}
              >
                {isRefreshing ? '🔄 Refreshing...' : '🔄 Refresh'}
              </button>
            </div>
          </div>
        </div>

        {/* Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="text-gray-400 text-sm mb-2">Total Packets</div>
            <div className="text-3xl font-bold">{statistics?.total_packets || 0}</div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="text-gray-400 text-sm mb-2">Attacks Detected</div>
            <div className="text-3xl font-bold text-red-500">{statistics?.attacks_detected || 0}</div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="text-gray-400 text-sm mb-2">Normal Traffic</div>
            <div className="text-3xl font-bold text-green-500">{statistics?.normal_packets || 0}</div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="text-gray-400 text-sm mb-2">Attack Rate</div>
            <div className="text-3xl font-bold text-yellow-500">
              {((statistics?.attack_rate || 0) * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Pie Chart */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-xl font-semibold mb-4">Traffic Distribution</h2>
            {pieData.length > 0 && pieData.some(d => d.value > 0) ? (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${((percent || 0) * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[300px] flex items-center justify-center text-gray-500">
                No data yet - send some packets to analyze!
              </div>
            )}
          </div>

          {/* Recent Activity */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-xl font-semibold mb-4">Recent Detections</h2>
            <div className="space-y-2 max-h-[300px] overflow-y-auto">
              {recentDetections.length > 0 ? (
                recentDetections.map((detection, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded ${
                      detection.is_attack ? 'bg-red-900/30 border border-red-700' : 'bg-green-900/30 border border-green-700'
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <div className="flex items-center gap-2">
                        <span className={`font-semibold ${
                          detection.is_attack ? 'text-red-400' : 'text-green-400'
                        }`}>
                          {detection.is_attack ? '🚨 Attack' : '✅ Normal'}
                        </span>
                        {detection.alert_level && detection.is_attack && (
                          <span className={`text-xs px-2 py-1 rounded ${
                            detection.alert_level === 'HIGH' ? 'bg-red-600' : 'bg-yellow-600'
                          }`}>
                            {detection.alert_level}
                          </span>
                        )}
                      </div>
                      <div className="text-sm text-gray-400">
                        {detection.predictions?.random_forest?.confidence 
                          ? `${(detection.predictions.random_forest.confidence * 100).toFixed(1)}%`
                          : 'N/A'}
                      </div>
                    </div>
                    <div className="text-xs text-gray-400 mt-1">{detection.reason}</div>
                    <div className="text-xs text-gray-500 mt-1">
                      {new Date(detection.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-gray-500 text-center py-8">No detections yet</div>
              )}
            </div>
          </div>
        </div>

        {/* Detection Log Table */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h2 className="text-xl font-semibold mb-4">Detection Log</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left border-b border-gray-700">
                  <th className="pb-2 text-gray-400">Timestamp</th>
                  <th className="pb-2 text-gray-400">Status</th>
                  <th className="pb-2 text-gray-400">Alert Level</th>
                  <th className="pb-2 text-gray-400">Reason</th>
                  <th className="pb-2 text-gray-400">RF Confidence</th>
                  <th className="pb-2 text-gray-400">ISO Score</th>
                </tr>
              </thead>
              <tbody>
                {recentDetections.length > 0 ? (
                  recentDetections.map((detection, idx) => (
                    <tr key={idx} className="border-b border-gray-700">
                      <td className="py-2 text-sm text-gray-400">
                        {new Date(detection.timestamp).toLocaleTimeString()}
                      </td>
                      <td className="py-2">
                        <span className={`px-2 py-1 rounded text-xs ${
                          detection.is_attack
                            ? 'bg-red-900 text-red-200'
                            : 'bg-green-900 text-green-200'
                        }`}>
                          {detection.is_attack ? 'Attack' : 'Normal'}
                        </span>
                      </td>
                      <td className="py-2 text-sm">{detection.alert_level || 'N/A'}</td>
                      <td className="py-2 text-sm">{detection.reason}</td>
                      <td className="py-2 text-sm">
                        {detection.predictions?.random_forest?.confidence 
                          ? `${(detection.predictions.random_forest.confidence * 100).toFixed(1)}%`
                          : 'N/A'}
                      </td>
                      <td className="py-2 text-sm">
                        {detection.predictions?.isolation_forest?.anomaly_score?.toFixed(3) || 'N/A'}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={6} className="py-8 text-center text-gray-500">
                      No detections yet - waiting for traffic analysis...
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-gray-500 text-sm">
          <p>ML-Based IDS • Random Forest (76.77% accuracy) + Isolation Forest</p>
        </div>
      </div>
    </div>
  );
}