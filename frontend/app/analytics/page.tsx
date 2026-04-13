'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import axios from 'axios';
import Link from 'next/link';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Analytics() {
  const [hourlyData, setHourlyData] = useState<any[]>([]);
  const [attackTypes, setAttackTypes] = useState<any[]>([]);
  const [timeline, setTimeline] = useState<any[]>([]);

  const fetchAnalytics = async () => {
    try {
      const [hourly, types, time] = await Promise.all([
        axios.get(`${API_URL}/analytics/hourly?hours=168`),
        axios.get(`${API_URL}/analytics/attack-types`),
        axios.get(`${API_URL}/analytics/timeline?limit=50`)
      ]);

      setHourlyData(hourly.data.data);
      setAttackTypes(types.data.data);
      setTimeline(time.data.data);
    } catch (error) {
      console.error('Error fetching analytics:', error);
    }
  };

  useEffect(() => {
    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-black text-green-400 p-6 font-mono crt-screen relative overflow-hidden">
      {/* Scanlines overlay */}
      <div className="scanlines pointer-events-none"></div>
      
      {/* CRT glow overlay */}
      <div className="crt-glow pointer-events-none"></div>

      <div className="max-w-[95%] mx-auto relative z-10">
        {/* Header */}
        <div className="mb-6 border-4 border-green-400 p-4 bg-black/80">
          <div className="flex items-center justify-between">
            <h1 className="text-6xl font-bold tracking-wider glow-text">
              [[ ANALYTICS DASHBOARD ]]
            </h1>
            <Link 
              href="/"
              className="px-6 py-3 border-2 border-green-400 bg-black hover:bg-green-400 hover:text-black transition-all text-lg tracking-wider"
            >
              &lt;&lt; BACK TO MONITOR
            </Link>
          </div>
        </div>

        {/* Weekly Stats Chart */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="border-4 border-green-400 p-4 mb-6 bg-black/80"
        >
          <h2 className="text-3xl mb-4 tracking-wider">╔═══ DETECTIONS BY DAY (LAST 7 DAYS) ═══╗</h2>
          {hourlyData.length > 0 ? (
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={hourlyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#00ff41" opacity={0.1} />
                <XAxis 
                  dataKey="date" 
                  stroke="#00ff41"
                  tick={{ fill: '#00ff41', fontSize: 14 }}
                  angle={-45}
                  textAnchor="end"
                  height={100}
                />
                <YAxis stroke="#00ff41" tick={{ fill: '#00ff41', fontSize: 14 }} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#000', 
                    border: '2px solid #00ff41',
                    color: '#00ff41',
                    fontFamily: 'VT323, monospace',
                    fontSize: '16px'
                  }}
                />
                <Legend wrapperStyle={{ color: '#00ff41', fontFamily: 'VT323, monospace', fontSize: '18px' }} />
                <Line type="monotone" dataKey="attacks" stroke="#ff0040" strokeWidth={3} name="ATTACKS" />
                <Line type="monotone" dataKey="normal" stroke="#00ff41" strokeWidth={3} name="NORMAL" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[350px] flex items-center justify-center opacity-60 text-xl">
              NO DATA AVAILABLE // GENERATE TRAFFIC TO SEE ANALYTICS
            </div>
          )}
        </motion.div>

        {/* Attack Types Breakdown */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="border-4 border-green-400 p-4 mb-6 bg-black/80"
        >
          <h2 className="text-3xl mb-4 tracking-wider">╔═══ ATTACK TYPE DISTRIBUTION ═══╗</h2>
          {attackTypes.length > 0 ? (
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={attackTypes}>
                <CartesianGrid strokeDasharray="3 3" stroke="#00ff41" opacity={0.1} />
                <XAxis 
                  dataKey="type" 
                  stroke="#00ff41"
                  tick={{ fill: '#00ff41', fontSize: 16 }}
                />
                <YAxis stroke="#00ff41" tick={{ fill: '#00ff41', fontSize: 14 }} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#000', 
                    border: '2px solid #00ff41',
                    color: '#00ff41',
                    fontFamily: 'VT323, monospace',
                    fontSize: '16px'
                  }}
                />
                <Bar dataKey="count" fill="#ff0040" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[350px] flex items-center justify-center opacity-60 text-xl">
              NO ATTACK DATA // INJECT ATTACK VECTORS TO SEE BREAKDOWN
            </div>
          )}
        </motion.div>

        {/* Detection Timeline */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="border-4 border-green-400 p-4 bg-black/80"
        >
          <h2 className="text-3xl mb-4 tracking-wider">╔═══ DETECTION TIMELINE (LAST 50) ═══╗</h2>
          {timeline.length > 0 ? (
            <div className="space-y-2 max-h-[400px] overflow-y-auto custom-scrollbar">
              {timeline.map((item, idx) => (
                <div 
                  key={idx}
                  className={`p-3 border ${
                    item.is_attack ? 'border-red-500 bg-red-500/10' : 'border-green-400/30 bg-green-400/5'
                  }`}
                >
                  <div className="flex justify-between text-lg">
                    <span className={item.is_attack ? 'text-red-500' : 'text-green-400'}>
                      {item.is_attack ? '[!] ATTACK' : '[✓] NORMAL'}
                      {item.attack_type && ` - ${item.attack_type.toUpperCase()}`}
                    </span>
                    <span className="opacity-60">
                      {new Date(item.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="h-[400px] flex items-center justify-center opacity-60 text-xl">
              NO TIMELINE DATA AVAILABLE
            </div>
          )}
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
      `}</style>
    </div>
  );
}