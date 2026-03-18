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
  attack_type?: string;  
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

// Test with sample traffic - randomized
  const sendNormalTraffic = async () => {
    const normalPacket = {
      duration: Math.floor(Math.random() * 100),
      protocol_type: ['tcp', 'udp'][Math.floor(Math.random() * 2)],
      service: ['http', 'ftp', 'smtp', 'ssh'][Math.floor(Math.random() * 4)],
      flag: 'SF',
      src_bytes: Math.floor(Math.random() * 5000) + 100,
      dst_bytes: Math.floor(Math.random() * 10000) + 500,
      land: 0,
      wrong_fragment: 0,
      urgent: 0,
      hot: 0,
      num_failed_logins: 0,
      logged_in: Math.random() > 0.3 ? 1 : 0,
      num_compromised: 0,
      root_shell: 0,
      su_attempted: 0,
      num_root: 0,
      num_file_creations: 0,
      num_shells: 0,
      num_access_files: 0,
      num_outbound_cmds: 0,
      is_host_login: 0,
      is_guest_login: 0,
      count: Math.floor(Math.random() * 20) + 1,
      srv_count: Math.floor(Math.random() * 20) + 1,
      serror_rate: Math.random() * 0.1,
      srv_serror_rate: Math.random() * 0.1,
      rerror_rate: Math.random() * 0.1,
      srv_rerror_rate: Math.random() * 0.1,
      same_srv_rate: 0.8 + Math.random() * 0.2,
      diff_srv_rate: Math.random() * 0.2,
      srv_diff_host_rate: Math.random() * 0.2,
      dst_host_count: Math.floor(Math.random() * 50) + 1,
      dst_host_srv_count: Math.floor(Math.random() * 50) + 1,
      dst_host_same_srv_rate: 0.7 + Math.random() * 0.3,
      dst_host_diff_srv_rate: Math.random() * 0.3,
      dst_host_same_src_port_rate: Math.random() * 0.5,
      dst_host_srv_diff_host_rate: Math.random() * 0.2,
      dst_host_serror_rate: Math.random() * 0.1,
      dst_host_srv_serror_rate: Math.random() * 0.1,
      dst_host_rerror_rate: Math.random() * 0.1,
      dst_host_srv_rerror_rate: Math.random() * 0.1
    };

    try {
      await axios.post(`${API_URL}/detect`, normalPacket);
      await handleRefresh();
       // Clear detection history
  const handleClear = async () => {
    try {
      await axios.post(`${API_URL}/clear`);
      await handleRefresh();
    } catch (error) {
      console.error('Error clearing history:', error);
    }
  };
    } catch (error) {
      console.error('Error sending normal traffic:', error);
    }
  };

  const sendAttackTraffic = async () => {
    // Random attack types
    const attackTypes = [
      // Port Scan
      {
        name: 'Port Scan',
        protocol_type: 'tcp',
        service: 'private',
        flag: 'REJ',
        src_bytes: 0,
        dst_bytes: 0,
        count: Math.floor(Math.random() * 300) + 200,
        srv_count: Math.floor(Math.random() * 300) + 200,
        serror_rate: 0.9 + Math.random() * 0.1,
        srv_serror_rate: 0.9 + Math.random() * 0.1,
        dst_host_count: Math.floor(Math.random() * 100) + 150,
        dst_host_serror_rate: 0.9 + Math.random() * 0.1
      },
      // DoS Attack
      {
        name: 'DoS Attack',
        protocol_type: 'tcp',
        service: 'http',
        flag: 'S0',
        src_bytes: Math.floor(Math.random() * 1000),
        dst_bytes: 0,
        count: Math.floor(Math.random() * 500) + 400,
        srv_count: Math.floor(Math.random() * 500) + 400,
        serror_rate: 0.8 + Math.random() * 0.2,
        srv_serror_rate: 0.8 + Math.random() * 0.2,
        dst_host_count: Math.floor(Math.random() * 200) + 100,
        dst_host_serror_rate: 0.7 + Math.random() * 0.3
      },
      // Brute Force
      {
        name: 'Brute Force',
        protocol_type: 'tcp',
        service: 'ftp',
        flag: 'SF',
        src_bytes: Math.floor(Math.random() * 500) + 50,
        dst_bytes: Math.floor(Math.random() * 500) + 50,
        count: Math.floor(Math.random() * 100) + 50,
        srv_count: Math.floor(Math.random() * 100) + 50,
        serror_rate: 0.3 + Math.random() * 0.4,
        srv_serror_rate: 0.3 + Math.random() * 0.4,
        dst_host_count: Math.floor(Math.random() * 50) + 20,
        dst_host_serror_rate: 0.4 + Math.random() * 0.3
      },
      // IP Sweep
      {
        name: 'IP Sweep',
        protocol_type: 'icmp',
        service: 'eco_i',
        flag: 'SF',
        src_bytes: 8,
        dst_bytes: 0,
        count: Math.floor(Math.random() * 400) + 300,
        srv_count: Math.floor(Math.random() * 400) + 300,
        serror_rate: 0,
        srv_serror_rate: 0,
        dst_host_count: Math.floor(Math.random() * 200) + 200,
        dst_host_serror_rate: 0
      }
    ];

    // Pick random attack
    const attack = attackTypes[Math.floor(Math.random() * attackTypes.length)];

    const attackPacket = {
      duration: Math.floor(Math.random() * 10),
      protocol_type: attack.protocol_type,
      service: attack.service,
      flag: attack.flag,
      src_bytes: attack.src_bytes,
      dst_bytes: attack.dst_bytes,
      land: 0,
      wrong_fragment: 0,
      urgent: 0,
      hot: Math.floor(Math.random() * 3),
      num_failed_logins: Math.floor(Math.random() * 5),
      logged_in: 0,
      num_compromised: Math.floor(Math.random() * 2),
      root_shell: Math.floor(Math.random() * 2),
      su_attempted: Math.floor(Math.random() * 2),
      num_root: Math.floor(Math.random() * 3),
      num_file_creations: 0,
      num_shells: 0,
      num_access_files: 0,
      num_outbound_cmds: 0,
      is_host_login: 0,
      is_guest_login: 0,
      count: attack.count,
      srv_count: attack.srv_count,
      serror_rate: attack.serror_rate,
      srv_serror_rate: attack.srv_serror_rate,
      rerror_rate: Math.random() * 0.2,
      srv_rerror_rate: Math.random() * 0.2,
      same_srv_rate: 0.8 + Math.random() * 0.2,
      diff_srv_rate: Math.random() * 0.2,
      srv_diff_host_rate: Math.random() * 0.3,
      dst_host_count: attack.dst_host_count,
      dst_host_srv_count: Math.floor(Math.random() * 200) + 100,
      dst_host_same_srv_rate: 0.5 + Math.random() * 0.5,
      dst_host_diff_srv_rate: Math.random() * 0.4,
      dst_host_same_src_port_rate: Math.random() * 0.3,
      dst_host_srv_diff_host_rate: Math.random() * 0.3,
      dst_host_serror_rate: attack.dst_host_serror_rate,
      dst_host_srv_serror_rate: 0.5 + Math.random() * 0.5,
      dst_host_rerror_rate: Math.random() * 0.3,
      dst_host_srv_rerror_rate: Math.random() * 0.3
    };

    try {
      await axios.post(`${API_URL}/detect`, attackPacket);
      await handleRefresh();
    } catch (error) {
      console.error('Error sending attack traffic:', error);
    }
  };

  // Manual refresh both
  const handleRefresh = async () => {
    setIsRefreshing(true);
    await Promise.all([fetchStats(), fetchDetections()]);
    setIsRefreshing(false);
  };
// Clear data
  const handleClear = async () => {
    try {
      await axios.post(`${API_URL}/clear`);
      await handleRefresh();
    } catch (error) {
      console.error('Error clearing history:', error);
    }
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
                onClick={handleClear}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg font-semibold transition-colors"
              >
                 Clear History
              </button>
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

        {/* Test Traffic Panel */}
        <div className="bg-gray-800 rounded-lg p-6 mb-8 border border-gray-700">
          <h2 className="text-xl font-semibold mb-4">🧪 Test Traffic Generator</h2>
          <p className="text-gray-400 text-sm mb-4">
            Send sample packets to test the detection system. Click buttons below to simulate normal traffic or attacks.
          </p>
          <div className="flex gap-4">
            <button
              onClick={sendNormalTraffic}
              className="px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg font-semibold transition-colors"
            >
              ✅ Send Normal Traffic
            </button>
            <button
              onClick={sendAttackTraffic}
              className="px-6 py-3 bg-red-600 hover:bg-red-700 rounded-lg font-semibold transition-colors"
            >
              🚨 Send Attack Traffic (Port Scan)
            </button>
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
                        {/* Show attack type */}
                        {detection.attack_type && (
                          <span className="text-xs px-2 py-1 rounded bg-orange-600">
                            {detection.attack_type}
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
                  <th className="pb-2 text-gray-400">Attack Type</th>
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
                      <td className="py-2 text-sm">
                        {detection.attack_type ? (
                          <span className="px-2 py-1 rounded text-xs bg-orange-900 text-orange-200">
                            {detection.attack_type}
                          </span>
                        ) : (
                          <span className="text-gray-500">-</span>
                        )}
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
                    <td colSpan={7} className="py-8 text-center text-gray-500">
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