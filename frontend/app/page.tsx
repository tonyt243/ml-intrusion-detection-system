'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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

  const fetchDetections = async () => {
    try {
      const response = await axios.get(`${API_URL}/recent?limit=10`);
      setRecentDetections(response.data.recent_detections || []);
    } catch (error) {
      console.error('Error fetching detections:', error);
    }
  };

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
    } catch (error) {
      console.error('Error sending normal traffic:', error);
    }
  };

  const sendAttackTraffic = async () => {
    const attackTypes = [
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

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await Promise.all([fetchStats(), fetchDetections()]);
    setIsRefreshing(false);
  };

  const handleClear = async () => {
    try {
      await axios.post(`${API_URL}/clear`);
      await handleRefresh();
    } catch (error) {
      console.error('Error clearing history:', error);
    }
  };

  useEffect(() => {
    fetchStats();
    fetchDetections();
  }, []);

  const pieData = statistics ? [
    { name: 'Normal', value: statistics.normal_packets, color: '#10b981' },
    { name: 'Attacks', value: statistics.attacks_detected, color: '#ef4444' }
  ] : [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Animated Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-5xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                Aurelius - Intrusion Detection System
              </h1>
              <p className="text-gray-400">Real-time network traffic analysis using machine learning</p>
            </div>
            <div className="flex items-center gap-4">
              <motion.div 
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.3 }}
                className="flex items-center gap-2"
              >
                <motion.div 
                  animate={{ 
                    scale: isConnected ? [1, 1.2, 1] : 1,
                    boxShadow: isConnected ? ['0 0 0px #10b981', '0 0 20px #10b981', '0 0 0px #10b981'] : 'none'
                  }}
                  transition={{ repeat: Infinity, duration: 2 }}
                  className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}
                />
                <span className="text-sm text-gray-400">{isConnected ? 'Connected' : 'Disconnected'}</span>
              </motion.div>
              
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleClear}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg font-semibold transition-all shadow-lg hover:shadow-xl"
              >
                🗑️ Clear
              </motion.button>
              
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleRefresh}
                disabled={isRefreshing}
                className={`px-4 py-2 rounded-lg font-semibold transition-all shadow-lg hover:shadow-xl ${
                  isRefreshing
                    ? 'bg-gray-700 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700'
                }`}
              >
                <motion.span
                  animate={isRefreshing ? { rotate: 360 } : {}}
                  transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                  className="inline-block"
                >
                  🔄
                </motion.span> {isRefreshing ? 'Refreshing...' : 'Refresh'}
              </motion.button>
            </div>
          </div>
        </motion.div>

        {/* Test Traffic Panel */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 mb-8 border border-gray-700/50 shadow-2xl"
        >
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <span className="text-2xl">Test Traffic Generator</span>
          </h2>
          <p className="text-gray-400 text-sm mb-4">
            Send sample packets to test the detection system
          </p>
          <div className="flex gap-4">
            <motion.button
              whileHover={{ scale: 1.05, boxShadow: "0 0 25px rgba(16, 185, 129, 0.5)" }}
              whileTap={{ scale: 0.95 }}
              onClick={sendNormalTraffic}
              className="px-6 py-3 bg-gradient-to-r from-green-600 to-green-500 hover:from-green-500 hover:to-green-400 rounded-lg font-semibold transition-all shadow-lg"
            >
              ✅ Send Normal Traffic
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05, boxShadow: "0 0 25px rgba(239, 68, 68, 0.5)" }}
              whileTap={{ scale: 0.95 }}
              onClick={sendAttackTraffic}
              className="px-6 py-3 bg-gradient-to-r from-red-600 to-red-500 hover:from-red-500 hover:to-red-400 rounded-lg font-semibold transition-all shadow-lg"
            >
              🚨 Send Attack Traffic
            </motion.button>
          </div>
        </motion.div>

        {/* Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          {[
            { label: 'Total Packets', value: statistics?.total_packets || 0, color: 'from-blue-600 to-blue-400', delay: 0.3 },
            { label: 'Attacks Detected', value: statistics?.attacks_detected || 0, color: 'from-red-600 to-red-400', delay: 0.4 },
            { label: 'Normal Traffic', value: statistics?.normal_packets || 0, color: 'from-green-600 to-green-400', delay: 0.5 },
            { label: 'Attack Rate', value: `${((statistics?.attack_rate || 0) * 100).toFixed(1)}%`, color: 'from-yellow-600 to-yellow-400', delay: 0.6 }
          ].map((stat, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: stat.delay, duration: 0.5 }}
              whileHover={{ y: -5, boxShadow: "0 10px 40px rgba(0,0,0,0.3)" }}
              className={`bg-gradient-to-br ${stat.color} rounded-xl p-6 shadow-xl border border-white/10`}
            >
              <div className="text-white/80 text-sm mb-2">{stat.label}</div>
              <motion.div 
                key={stat.value}
                initial={{ scale: 1.2, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="text-4xl font-bold text-white"
              >
                {stat.value}
              </motion.div>
            </motion.div>
          ))}
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Pie Chart */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.7, duration: 0.6 }}
            className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50 shadow-2xl"
          >
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
                    animationBegin={0}
                    animationDuration={800}
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
          </motion.div>

          {/* Recent Activity */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.7, duration: 0.6 }}
            className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50 shadow-2xl"
          >
            <h2 className="text-xl font-semibold mb-4">Recent Detections</h2>
            <div className="space-y-2 max-h-[300px] overflow-y-auto custom-scrollbar">
              <AnimatePresence>
                {recentDetections.length > 0 ? (
                  recentDetections.map((detection, idx) => (
                    <motion.div
                      key={`${detection.timestamp}-${idx}`}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 20 }}
                      transition={{ delay: idx * 0.05 }}
                      whileHover={{ scale: 1.02, x: 5 }}
                      className={`p-3 rounded-lg ${
                        detection.is_attack 
                          ? 'bg-red-900/30 border border-red-700/50 hover:border-red-600' 
                          : 'bg-green-900/30 border border-green-700/50 hover:border-green-600'
                      } transition-all`}
                    >
                      <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2">
                          <motion.span 
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            className={`font-semibold ${
                              detection.is_attack ? 'text-red-400' : 'text-green-400'
                            }`}
                          >
                            {detection.is_attack ? '🚨 Attack' : '✅ Normal'}
                          </motion.span>
                          {detection.alert_level && detection.is_attack && (
                            <motion.span 
                              initial={{ scale: 0 }}
                              animate={{ scale: 1 }}
                              className={`text-xs px-2 py-1 rounded ${
                                detection.alert_level === 'HIGH' ? 'bg-red-600' : 'bg-yellow-600'
                              }`}
                            >
                              {detection.alert_level}
                            </motion.span>
                          )}
                          {detection.attack_type && (
                            <motion.span 
                              initial={{ scale: 0, rotate: -10 }}
                              animate={{ scale: 1, rotate: 0 }}
                              className="text-xs px-2 py-1 rounded bg-orange-600"
                            >
                              {detection.attack_type}
                            </motion.span>
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
                    </motion.div>
                  ))
                ) : (
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-gray-500 text-center py-8"
                  >
                    No detections yet
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        </div>

        {/* Detection Log Table */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9, duration: 0.6 }}
          className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50 shadow-2xl"
        >
          <h2 className="text-xl font-semibold mb-4">Detection Log</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left border-b border-gray-700">
                  <th className="pb-3 text-gray-400 font-semibold">Timestamp</th>
                  <th className="pb-3 text-gray-400 font-semibold">Status</th>
                  <th className="pb-3 text-gray-400 font-semibold">Attack Type</th>
                  <th className="pb-3 text-gray-400 font-semibold">Alert Level</th>
                  <th className="pb-3 text-gray-400 font-semibold">Reason</th>
                  <th className="pb-3 text-gray-400 font-semibold">RF Confidence</th>
                  <th className="pb-3 text-gray-400 font-semibold">ISO Score</th>
                </tr>
              </thead>
              <tbody>
                <AnimatePresence>
                  {recentDetections.length > 0 ? (
                    recentDetections.map((detection, idx) => (
                      <motion.tr 
                        key={`${detection.timestamp}-${idx}`}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ delay: idx * 0.03 }}
                        whileHover={{ backgroundColor: 'rgba(255,255,255,0.05)' }}
                        className="border-b border-gray-700/50 transition-colors"
                      >
                        <td className="py-3 text-sm text-gray-400">
                          {new Date(detection.timestamp).toLocaleTimeString()}
                        </td>
                        <td className="py-3">
                          <motion.span 
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            className={`px-2 py-1 rounded text-xs ${
                              detection.is_attack
                                ? 'bg-red-900 text-red-200'
                                : 'bg-green-900 text-green-200'
                            }`}
                          >
                            {detection.is_attack ? 'Attack' : 'Normal'}
                          </motion.span>
                        </td>
                        <td className="py-3 text-sm">
                          {detection.attack_type ? (
                            <motion.span 
                              initial={{ scale: 0 }}
                              animate={{ scale: 1 }}
                              className="px-2 py-1 rounded text-xs bg-orange-900 text-orange-200"
                            >
                              {detection.attack_type}
                            </motion.span>
                          ) : (
                            <span className="text-gray-500">-</span>
                          )}
                        </td>
                        <td className="py-3 text-sm">{detection.alert_level || 'N/A'}</td>
                        <td className="py-3 text-sm">{detection.reason}</td>
                        <td className="py-3 text-sm">
                          {detection.predictions?.random_forest?.confidence 
                            ? `${(detection.predictions.random_forest.confidence * 100).toFixed(1)}%`
                            : 'N/A'}
                        </td>
                        <td className="py-3 text-sm">
                          {detection.predictions?.isolation_forest?.anomaly_score?.toFixed(3) || 'N/A'}
                        </td>
                      </motion.tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={7} className="py-8 text-center text-gray-500">
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                        >
                          No detections yet - waiting for traffic analysis...
                        </motion.div>
                      </td>
                    </tr>
                  )}
                </AnimatePresence>
              </tbody>
            </table>
          </div>
        </motion.div>

        {/* Footer */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
          className="mt-8 text-center text-gray-500 text-sm"
        >
          <p className="mb-1">ML-Based IDS • Random Forest (76.77% accuracy) + Isolation Forest</p>
        </motion.div>
      </div>

      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(0, 0, 0, 0.2);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.2);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.3);
        }
      `}</style>
    </div>
  );
}