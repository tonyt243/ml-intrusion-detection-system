'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import axios from 'axios';
import Link from 'next/link';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Detection {
  id: number;
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

interface DetectionDetail {
  id: number;
  timestamp: string;
  is_attack: boolean;
  alert_level: string;
  reason: string;
  attack_type: string | null;
  predictions: {
    random_forest: {
      confidence: number;
      attack_probability: number;
      verdict: string;
    };
    isolation_forest: {
      anomaly_score: number;
      verdict: string;
    };
  };
  key_features: {
    duration: number;
    protocol_type: string;
    service: string;
    flag: string;
    src_bytes: number;
    dst_bytes: number;
    count: number;
    srv_count: number;
    dst_host_count: number;
    serror_rate: number;
    srv_serror_rate: number;
    dst_host_serror_rate: number;
    same_srv_rate: number;
    dst_host_same_srv_rate: number;
    logged_in: number;
    num_failed_logins: number;
    root_shell: number;
    num_compromised: number;
  };
}

interface Statistics {
  total_packets: number;
  attacks_detected: number;
  normal_packets: number;
  attack_rate: number;
}

// Live clock component
const LiveClock = () => {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="text-base opacity-80">
      {time.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit',
        hour12: true 
      })}
    </div>
  );
};

export default function Dashboard() {
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [recentDetections, setRecentDetections] = useState<Detection[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [bootSequence, setBootSequence] = useState(true);
  const [selectedDetection, setSelectedDetection] = useState<DetectionDetail | null>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);

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

  const fetchDetectionDetail = async (detectionId: number) => {
    try {
      const response = await axios.get(`${API_URL}/detection/${detectionId}`);
      setSelectedDetection(response.data);
      setShowDetailModal(true);
    } catch (error) {
      console.error('Error fetching detection detail:', error);
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
    // Boot sequence
    const timer = setTimeout(() => setBootSequence(false), 2000);
    fetchStats();
    fetchDetections();
    return () => clearTimeout(timer);
  }, []);

  const pieData = statistics ? [
    { name: 'NORMAL', value: statistics.normal_packets, color: '#00ff41' },
    { name: 'THREAT', value: statistics.attacks_detected, color: '#ff0040' }
  ] : [];

  if (bootSequence) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="font-mono text-green-400 text-2xl mb-4"
          >
            <div className="mb-2">INITIALIZING DETECTION SYSTEM...</div>
            <div className="mb-2">LOADING ML-IDS v1.0...</div>
            <div className="mb-2">ESTABLISHING SECURE CONNECTION...</div>
            <motion.div
              animate={{ opacity: [1, 0, 1] }}
              transition={{ repeat: Infinity, duration: 1 }}
            >
              &#9608;
            </motion.div>
          </motion.div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-green-400 p-6 font-mono crt-screen relative overflow-hidden">
      {/* Scanlines overlay */}
      <div className="scanlines pointer-events-none"></div>
      
      {/* CRT glow overlay */}
      <div className="crt-glow pointer-events-none"></div>

      <div className="max-w-[95%] mx-auto relative z-10">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
          className="mb-6 border-4 border-green-400 p-4 bg-black/80"
        >
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-6xl font-bold mb-2 tracking-wider glow-text">
                [[ AURELIUS | ML-INTRUSION DETECTION SYSTEM ]]
              </h1>
              <p className="text-lg opacity-80">NETWORK SECURITY TERMINAL // REAL-TIME THREAT ANALYSIS</p>
              <p className="text-lg opacity-80 mt-1">SYSTEM ID: ML-IDS-1.0.4 // KERNEL: RANDOM_FOREST + ISOLATION_FOREST</p>
            </div>
            <div className="flex flex-col items-end gap-2">
              <div className="flex items-center gap-2">
                <motion.div 
                  animate={{ opacity: [1, 0.3, 1] }}
                  transition={{ repeat: Infinity, duration: 1.5 }}
                  className={`w-3 h-3 border-2 ${isConnected ? 'border-green-400 bg-green-400' : 'border-red-500 bg-red-500'}`}
                />
                <span className="text-s tracking-wider">{isConnected ? '[ONLINE]' : '[OFFLINE]'}</span>
              </div>
              {/* Live Clock */}
              <LiveClock /> 
            </div>
          </div>

          <div className="flex gap-2">
            <button
              onClick={handleClear}
              className="px-6 py-3 border-2 border-green-400 bg-black hover:bg-green-400 hover:text-black transition-all text-lg tracking-wider"
            >
              [CLEAR LOG]
            </button>
            <button
              onClick={handleRefresh}
              disabled={isRefreshing}
              className="px-6 py-3 border-2 border-green-400 bg-black hover:bg-green-400 hover:text-black transition-all text-lg tracking-wider disabled:opacity-50"
            >
              {isRefreshing ? '[SCANNING...]' : '[REFRESH]'}
            </button>
            <Link
              href="/analytics"
              className="px-6 py-3 border-2 border-amber-400 bg-black hover:bg-amber-400 hover:text-black transition-all text-lg tracking-wider text-amber-400 inline-block"
            >
              [ANALYTICS]
            </Link>
            <Link
              href="/capture"
              className="px-6 py-3 border-2 border-cyan-400 bg-black hover:bg-cyan-400 hover:text-black transition-all text-lg tracking-wider text-cyan-400 inline-block"
            >
              [LIVE CAPTURE]
            </Link>
          </div>
        </motion.div>

        {/* Test Panel */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="border-4 border-green-400 p-4 mb-6 bg-black/80"
        >
          <h2 className="text-3xl mb-4 tracking-wider">╔═══ PACKET INJECTION TOOL ═══╗</h2>
          <div className="flex gap-3">
            <button
              onClick={sendNormalTraffic}
              className="px-6 py-3 border-2 border-green-400 bg-black hover:bg-green-400 hover:text-black transition-all tracking-wider"
            >
              &gt;&gt; SEND NORMAL PACKET
            </button>
            <button
              onClick={sendAttackTraffic}
              className="px-6 py-3 border-2 border-red-500 bg-black hover:bg-red-500 hover:text-black transition-all tracking-wider text-red-500"
            >
              &gt;&gt; INJECT ATTACK VECTOR
            </button>
          </div>
        </motion.div>

        {/* Stats Grid */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          {[
            { label: 'TOTAL PACKETS', value: statistics?.total_packets || 0, symbol: '█' },
            { label: 'THREATS DETECTED', value: statistics?.attacks_detected || 0, symbol: '▲', danger: true },
            { label: 'SAFE PACKETS', value: statistics?.normal_packets || 0, symbol: '●' },
            { label: 'THREAT RATIO', value: `${((statistics?.attack_rate || 0) * 100).toFixed(1)}%`, symbol: '◆', danger: (statistics?.attack_rate || 0) > 0.5 }
          ].map((stat, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 + idx * 0.05 }}
              className={`border-2 ${stat.danger ? 'border-red-500' : 'border-green-400'} p-4 bg-black/80`}
            >
              <div className={`text-base mb-2 ${stat.danger ? 'text-red-500' : 'opacity-60'}`}>{stat.label}</div>
              <div className={`text-6xl font-bold ${stat.danger ? 'text-red-500 glow-text-red' : 'glow-text'}`}>
                {stat.symbol} {stat.value}
              </div>
            </motion.div>
          ))}
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-2 gap-6 mb-6">
          {/* Pie Chart */}
          <div className="border-4 border-green-400 p-4 bg-black/80">
            <h2 className="text-3xl mb-4 tracking-wider">╔═══ TRAFFIC ANALYSIS ═══╗</h2>
            {pieData.length > 0 && pieData.some(d => d.value > 0) ? (
              <div className="relative">
                <ResponsiveContainer width="100%" height={250}>
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
                      stroke="#00ff41"
                      strokeWidth={2}
                    >
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#000', 
                        border: '2px solid #00ff41',
                        color: '#00ff41',
                        fontFamily: 'monospace'
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <div className="text-center text-xs opacity-60 mt-2">
                  DISTRIBUTION MATRIX // {statistics?.total_packets || 0} SAMPLES
                </div>
              </div>
            ) : (
              <div className="h-[250px] flex items-center justify-center opacity-60">
                NO DATA AVAILABLE // AWAITING PACKETS...
              </div>
            )}
          </div>

          {/* Recent Activity Log */}
          <div className="border-4 border-green-400 p-4 bg-black/80">
            <h2 className="text-3xl mb-4 tracking-wider">╔═══ THREAT LOG ═══╗</h2>
            <div className="space-y-1 max-h-[250px] overflow-y-auto custom-scrollbar text-sm">
              {recentDetections.length > 0 ? (
                recentDetections.map((detection, idx) => (
                  <motion.div
                    key={`${detection.timestamp}-${idx}`}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.03 }}
                    className={`p-2 border ${
                      detection.is_attack 
                        ? 'border-red-500 bg-red-500/10' 
                        : 'border-green-400/30 bg-green-400/5'
                    }`}
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <span className={detection.is_attack ? 'text-red-500' : 'text-green-400'}>
                          {detection.is_attack ? '[!] THREAT' : '[✓] CLEAN'}
                        </span>
                        {detection.attack_type && (
                          <span className="ml-2 text-amber-400">
                            [{detection.attack_type.toUpperCase()}]
                          </span>
                        )}
                        {detection.alert_level && detection.is_attack && (
                          <span className="ml-2 text-red-500 text-xs">
                            {detection.alert_level}
                          </span>
                        )}
                      </div>
                      <div className="text-xs opacity-60">
                        {detection.predictions?.random_forest?.confidence 
                          ? `${(detection.predictions.random_forest.confidence * 100).toFixed(0)}%`
                          : 'N/A'}
                      </div>
                    </div>
                    <div className="text-xs opacity-60 mt-1">
                      {(() => {
                        const utcDate = new Date(detection.timestamp.endsWith('Z') ? detection.timestamp : detection.timestamp + 'Z');
                        return utcDate.toLocaleTimeString('en-US', { 
                          hour: '2-digit', 
                          minute: '2-digit', 
                          second: '2-digit',
                          hour12: true
                        });
                      })()}
                    </div>
                  </motion.div>
                ))
              ) : (
                <div className="text-center opacity-60 py-8">
                  MONITORING... NO EVENTS LOGGED
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Detection Table */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="border-4 border-green-400 p-4 bg-black/80"
        >
          <h2 className="text-3xl mb-4 tracking-wider">╔═══ DETAILED DETECTION MATRIX ═══╗</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-lg">
              <thead>
                <tr className="border-b-2 border-green-400">
                  <th className="pb-2 text-left">TIMESTAMP</th>
                  <th className="pb-2 text-left">STATUS</th>
                  <th className="pb-2 text-left">VECTOR</th>
                  <th className="pb-2 text-left">ALERT</th>
                  <th className="pb-2 text-left">REASON</th>
                  <th className="pb-2 text-left">RF_CONF</th>
                  <th className="pb-2 text-left">ISO_SCORE</th>
                  <th className="pb-2 text-left">ACTION</th>
                </tr>
              </thead>
              <tbody>
                {recentDetections.length > 0 ? (
                  recentDetections.map((detection, idx) => (
                    <motion.tr 
                      key={`${detection.timestamp}-${idx}`}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: idx * 0.02 }}
                      className="border-b border-green-400/30 hover:bg-green-400/10"
                    >
                      <td className="py-2 opacity-60">
                        {(() => {
                          const utcDate = new Date(detection.timestamp.endsWith('Z') ? detection.timestamp : detection.timestamp + 'Z');
                          return utcDate.toLocaleTimeString('en-US', { 
                            hour: '2-digit', 
                            minute: '2-digit', 
                            second: '2-digit',
                            hour12: true
                          });
                        })()}
                      </td>
                      <td className="py-2">
                        <span className={detection.is_attack ? 'text-red-500' : 'text-green-400'}>
                          {detection.is_attack ? '[THREAT]' : '[SAFE]'}
                        </span>
                      </td>
                      <td className="py-2">
                        {detection.attack_type ? (
                          <span className="text-amber-400">{detection.attack_type.toUpperCase()}</span>
                        ) : (
                          <span className="opacity-40">-</span>
                        )}
                      </td>
                      <td className="py-2">{detection.alert_level || 'NONE'}</td>
                      <td className="py-2 opacity-80">{detection.reason}</td>
                      <td className="py-2">
                        {detection.predictions?.random_forest?.confidence 
                          ? `${(detection.predictions.random_forest.confidence * 100).toFixed(1)}%`
                          : 'N/A'}
                      </td>
                      <td className="py-2">
                        {detection.predictions?.isolation_forest?.anomaly_score?.toFixed(3) || 'N/A'}
                      </td>
                      <td className="py-2">
                        <button
                          onClick={() => fetchDetectionDetail(detection.id)}
                          className="px-3 py-1 border border-green-400 hover:bg-green-400 hover:text-black transition-all text-sm"
                        >
                          [DETAILS]
                        </button>
                      </td>
                    </motion.tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={8} className="py-8 text-center opacity-60">
                      NO DETECTIONS RECORDED // SYSTEM IDLE
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </motion.div>

        {/* Detail Modal */}
        <AnimatePresence>
          {showDetailModal && selectedDetection && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/90 flex items-center justify-center z-50 p-6"
              onClick={() => setShowDetailModal(false)}
            >
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                className="border-4 border-green-400 bg-black p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto custom-scrollbar"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex justify-between items-start mb-6">
                  <h2 className="text-4xl font-bold glow-text">
                    [[ DETECTION DETAILS ]]
                  </h2>
                  <button
                    onClick={() => setShowDetailModal(false)}
                    className="px-4 py-2 border-2 border-red-500 text-red-500 hover:bg-red-500 hover:text-black transition-all"
                  >
                    [CLOSE]
                  </button>
                </div>

                {/* Detection Status */}
                <div className="border-2 border-green-400 p-4 mb-4">
                  <h3 className="text-2xl mb-3">╔═══ STATUS ═══╗</h3>
                  <div className="grid grid-cols-2 gap-4 text-lg">
                    <div>
                      <span className="opacity-60">Verdict:</span>{' '}
                      <span className={selectedDetection.is_attack ? 'text-red-500' : 'text-green-400'}>
                        {selectedDetection.is_attack ? 'THREAT DETECTED' : 'NORMAL TRAFFIC'}
                      </span>
                    </div>
                    <div>
                      <span className="opacity-60">Alert Level:</span> {selectedDetection.alert_level}
                    </div>
                    <div>
                      <span className="opacity-60">Attack Type:</span>{' '}
                      <span className="text-amber-400">
                        {selectedDetection.attack_type || 'N/A'}
                      </span>
                    </div>
                    <div>
                      <span className="opacity-60">Timestamp:</span>{' '}
                      {new Date(selectedDetection.timestamp).toLocaleString()}
                    </div>
                  </div>
                </div>

                {/* Model Predictions */}
                <div className="border-2 border-green-400 p-4 mb-4">
                  <h3 className="text-2xl mb-3">╔═══ MODEL PREDICTIONS ═══╗</h3>
                  
                  {/* Random Forest */}
                  <div className="mb-4">
                    <div className="text-xl text-amber-400 mb-2">RANDOM FOREST CLASSIFIER:</div>
                    <div className="grid grid-cols-2 gap-4 text-lg ml-4">
                      <div>
                        <span className="opacity-60">Verdict:</span>{' '}
                        <span className={selectedDetection.predictions.random_forest.verdict === 'ATTACK' ? 'text-red-500' : 'text-green-400'}>
                          {selectedDetection.predictions.random_forest.verdict}
                        </span>
                      </div>
                      <div>
                        <span className="opacity-60">Confidence:</span>{' '}
                        {(selectedDetection.predictions.random_forest.confidence * 100).toFixed(1)}%
                      </div>
                      <div>
                        <span className="opacity-60">Attack Probability:</span>{' '}
                        {(selectedDetection.predictions.random_forest.attack_probability * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  {/* Isolation Forest */}
                  <div>
                    <div className="text-xl text-amber-400 mb-2">ISOLATION FOREST (ANOMALY DETECTION):</div>
                    <div className="grid grid-cols-2 gap-4 text-lg ml-4">
                      <div>
                        <span className="opacity-60">Verdict:</span>{' '}
                        <span className={selectedDetection.predictions.isolation_forest.verdict === 'ANOMALY' ? 'text-red-500' : 'text-green-400'}>
                          {selectedDetection.predictions.isolation_forest.verdict}
                        </span>
                      </div>
                      <div>
                        <span className="opacity-60">Anomaly Score:</span>{' '}
                        {selectedDetection.predictions.isolation_forest.anomaly_score.toFixed(3)}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Key Packet Features */}
<div className="border-2 border-green-400 p-4 mb-4">
  <h3 className="text-2xl mb-3">╔═══ KEY PACKET FEATURES ═══╗</h3>
  
  {/* Basic Connection Info */}
  <div className="mb-4">
    <h4 className="text-xl text-amber-400 mb-2">CONNECTION INFO:</h4>
    <div className="grid grid-cols-4 gap-3 text-base ml-4">
      <div>
        <span className="opacity-60">Duration:</span>{' '}
        <span className="text-green-400">{selectedDetection.key_features.duration}s</span>
      </div>
      <div>
        <span className="opacity-60">Protocol:</span>{' '}
        <span className="text-green-400">{selectedDetection.key_features.protocol_type.toUpperCase()}</span>
      </div>
      <div>
        <span className="opacity-60">Service:</span>{' '}
        <span className="text-green-400">{selectedDetection.key_features.service}</span>
      </div>
      <div>
        <span className="opacity-60">Flag:</span>{' '}
        <span className="text-green-400">{selectedDetection.key_features.flag}</span>
      </div>
    </div>
  </div>

  {/* Data Transfer */}
  <div className="mb-4">
    <h4 className="text-xl text-amber-400 mb-2">DATA TRANSFER:</h4>
    <div className="grid grid-cols-2 gap-3 text-base ml-4">
      <div>
        <span className="opacity-60">Source Bytes:</span>{' '}
        <span className="text-green-400">{selectedDetection.key_features.src_bytes.toLocaleString()}</span>
      </div>
      <div>
        <span className="opacity-60">Destination Bytes:</span>{' '}
        <span className="text-green-400">{selectedDetection.key_features.dst_bytes.toLocaleString()}</span>
      </div>
    </div>
  </div>

  {/* Connection Statistics */}
  <div className="mb-4">
    <h4 className="text-xl text-amber-400 mb-2">CONNECTION STATISTICS:</h4>
    <div className="grid grid-cols-3 gap-3 text-base ml-4">
      <div>
        <span className="opacity-60">Connection Count:</span>{' '}
        <span className="text-green-400">{selectedDetection.key_features.count}</span>
      </div>
      <div>
        <span className="opacity-60">Service Count:</span>{' '}
        <span className="text-green-400">{selectedDetection.key_features.srv_count}</span>
      </div>
      <div>
        <span className="opacity-60">Dest Host Count:</span>{' '}
        <span className="text-green-400">{selectedDetection.key_features.dst_host_count}</span>
      </div>
    </div>
  </div>

  {/* Error Rates */}
  <div className="mb-4">
    <h4 className="text-xl text-amber-400 mb-2">ERROR RATES:</h4>
    <div className="grid grid-cols-3 gap-3 text-base ml-4">
      <div>
        <span className="opacity-60">Service Error Rate:</span>{' '}
        <span className={selectedDetection.key_features.serror_rate > 0.5 ? 'text-red-500' : 'text-green-400'}>
          {(selectedDetection.key_features.serror_rate * 100).toFixed(1)}%
        </span>
      </div>
      <div>
        <span className="opacity-60">Srv Service Error:</span>{' '}
        <span className={selectedDetection.key_features.srv_serror_rate > 0.5 ? 'text-red-500' : 'text-green-400'}>
          {(selectedDetection.key_features.srv_serror_rate * 100).toFixed(1)}%
        </span>
      </div>
      <div>
        <span className="opacity-60">Dest Host Error:</span>{' '}
        <span className={selectedDetection.key_features.dst_host_serror_rate > 0.5 ? 'text-red-500' : 'text-green-400'}>
          {(selectedDetection.key_features.dst_host_serror_rate * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  </div>

  {/* Service Rates */}
  <div className="mb-4">
    <h4 className="text-xl text-amber-400 mb-2">SERVICE RATES:</h4>
    <div className="grid grid-cols-2 gap-3 text-base ml-4">
      <div>
        <span className="opacity-60">Same Service Rate:</span>{' '}
        <span className="text-green-400">{(selectedDetection.key_features.same_srv_rate * 100).toFixed(1)}%</span>
      </div>
      <div>
        <span className="opacity-60">Dest Same Service:</span>{' '}
        <span className="text-green-400">{(selectedDetection.key_features.dst_host_same_srv_rate * 100).toFixed(1)}%</span>
      </div>
    </div>
  </div>

  {/* Security Indicators */}
  <div>
    <h4 className="text-xl text-amber-400 mb-2">SECURITY INDICATORS:</h4>
    <div className="grid grid-cols-4 gap-3 text-base ml-4">
      <div>
        <span className="opacity-60">Logged In:</span>{' '}
        <span className={selectedDetection.key_features.logged_in === 1 ? 'text-green-400' : 'text-red-500'}>
          {selectedDetection.key_features.logged_in === 1 ? 'YES' : 'NO'}
        </span>
      </div>
      <div>
        <span className="opacity-60">Failed Logins:</span>{' '}
        <span className={selectedDetection.key_features.num_failed_logins > 0 ? 'text-red-500' : 'text-green-400'}>
          {selectedDetection.key_features.num_failed_logins}
        </span>
      </div>
      <div>
        <span className="opacity-60">Root Shell:</span>{' '}
        <span className={selectedDetection.key_features.root_shell > 0 ? 'text-red-500' : 'text-green-400'}>
          {selectedDetection.key_features.root_shell}
        </span>
      </div>
      <div>
        <span className="opacity-60">Compromised:</span>{' '}
        <span className={selectedDetection.key_features.num_compromised > 0 ? 'text-red-500' : 'text-green-400'}>
          {selectedDetection.key_features.num_compromised}
        </span>
      </div>
    </div>
  </div>
</div>

                {/* Reason */}
                <div className="border-2 border-green-400 p-4">
                  <h3 className="text-2xl mb-3">╔═══ ANALYSIS ═══╗</h3>
                  <div className="text-lg opacity-80">
                    {selectedDetection.reason}
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Footer */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="mt-6 text-center text-xs opacity-60 border-t-2 border-green-400/30 pt-4"
        >
          <p>RANDOM FOREST (76.77% ACCURACY) + ISOLATION FOREST</p>
        </motion.div>
      </div>

      {/* Retro CRT Styles */}
      <style jsx>{`
        @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');
        
        .crt-screen {
          font-family: 'VT323', monospace;
          font-size: 2.5rem;
          animation: flicker 0.15s infinite;
        }

        @keyframes flicker {
          0% { opacity: 0.97; }
          50% { opacity: 1; }
          100% { opacity: 0.97; }
        }

        .scanlines {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background: linear-gradient(
            to bottom,
            rgba(0, 0, 0, 0) 50%,
            rgba(0, 255, 65, 0.02) 50%
          );
          background-size: 100% 4px;
          z-index: 1;
          pointer-events: none;
          animation: scan 8s linear infinite;
        }

        @keyframes scan {
          0% { transform: translateY(0); }
          100% { transform: translateY(4px); }
        }

        .crt-glow {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background: radial-gradient(
            ellipse at center,
            rgba(0, 255, 65, 0.05) 0%,
            transparent 70%
          );
          z-index: 0;
        }

        .glow-text {
          text-shadow: 
            0 0 5px #00ff41,
            0 0 10px #00ff41;
          font-size: 1.2em;
        }

        .glow-text-red {
          text-shadow: 
            0 0 5px #ff0040,
            0 0 10px #ff0040;
          font-size: 1.2em;
        }

        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #000;
          border: 1px solid #00ff41;
        }
        
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #00ff41;
          border: 1px solid #000;
        }

        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #00ff88;
        }

        /* CRT curve effect */
        .crt-screen::before {
          content: "";
          display: block;
          position: absolute;
          top: 0;
          left: 0;
          bottom: 0;
          right: 0;
          background: 
            linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%),
            linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
          z-index: 2;
          background-size: 100% 2px, 3px 100%;
          pointer-events: none;
        }
      `}</style>
    </div>
  );
}