'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import Link from 'next/link';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface CapturedPacket {
  timestamp: string;
  src_ip: string;
  dst_ip: string;
  protocol: string;
  size: number;
  is_attack: boolean;
  alert_level: string;
  attack_type: string | null;
}

export default function LiveCapture() {
  const [isCapturing, setIsCapturing] = useState(false);
  const [packets, setPackets] = useState<CapturedPacket[]>([]);
  const [totalCaptured, setTotalCaptured] = useState(0);
  const [autoRefresh, setAutoRefresh] = useState(false);

  const checkStatus = async () => {
    try {
      const response = await axios.get(`${API_URL}/capture/status`);
      setIsCapturing(response.data.is_capturing);
      setTotalCaptured(response.data.packets_captured);
    } catch (error) {
      console.error('Error checking status:', error);
    }
  };

  const fetchPackets = async () => {
    try {
      const response = await axios.get(`${API_URL}/capture/packets?limit=50`);
      setPackets(response.data.packets.reverse()); // Show newest first
    } catch (error) {
      console.error('Error fetching packets:', error);
    }
  };

  const startCapture = async () => {
    try {
      await axios.post(`${API_URL}/capture/start`);
      setIsCapturing(true);
      setAutoRefresh(true);
    } catch (error) {
      console.error('Error starting capture:', error);
      alert('Failed to start capture. Make sure API is running as administrator.');
    }
  };

  const stopCapture = async () => {
    try {
      await axios.post(`${API_URL}/capture/stop`);
      setIsCapturing(false);
      setAutoRefresh(false);
    } catch (error) {
      console.error('Error stopping capture:', error);
    }
  };

  useEffect(() => {
    checkStatus();
    fetchPackets();
  }, []);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        fetchPackets();
        checkStatus();
      }, 2000); // Refresh every 2 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const attackCount = packets.filter(p => p.is_attack).length;
  const normalCount = packets.length - attackCount;

  return (
    <div className="min-h-screen bg-black text-green-400 p-6 font-mono crt-screen relative overflow-hidden">
      {/* Scanlines overlay */}
      <div className="scanlines pointer-events-none"></div>
      
      {/* CRT glow overlay */}
      <div className="crt-glow pointer-events-none"></div>

      <div className="max-w-[95%] mx-auto relative z-10">
        {/* Header */}
        <div className="mb-6 border-4 border-green-400 p-4 bg-black/80">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-6xl font-bold tracking-wider glow-text">
              [[ LIVE PACKET CAPTURE ]]
            </h1>
            <Link 
              href="/"
              className="px-6 py-3 border-2 border-green-400 bg-black hover:bg-green-400 hover:text-black transition-all text-lg tracking-wider"
            >
              &lt;&lt; BACK TO MONITOR
            </Link>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <motion.div 
                animate={{ opacity: isCapturing ? [1, 0.3, 1] : 1 }}
                transition={{ repeat: isCapturing ? Infinity : 0, duration: 1.5 }}
                className={`w-4 h-4 border-2 ${isCapturing ? 'border-red-500 bg-red-500' : 'border-green-400 bg-green-400'}`}
              />
              <span className="text-xl tracking-wider">
                {isCapturing ? '[CAPTURING...]' : '[IDLE]'}
              </span>
            </div>

            <div className="text-lg opacity-80">
              PACKETS: {totalCaptured}
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="border-4 border-green-400 p-4 mb-6 bg-black/80">
          <h2 className="text-3xl mb-4 tracking-wider">╔═══ CAPTURE CONTROLS ═══╗</h2>
          <div className="flex gap-3">
            {!isCapturing ? (
              <button
                onClick={startCapture}
                className="px-6 py-3 border-2 border-green-400 bg-black hover:bg-green-400 hover:text-black transition-all tracking-wider text-lg"
              >
                &gt;&gt; START CAPTURE
              </button>
            ) : (
              <button
                onClick={stopCapture}
                className="px-6 py-3 border-2 border-red-500 bg-black hover:bg-red-500 hover:text-black transition-all tracking-wider text-lg text-red-500"
              >
                &gt;&gt; STOP CAPTURE
              </button>
            )}
            <button
              onClick={fetchPackets}
              className="px-6 py-3 border-2 border-green-400 bg-black hover:bg-green-400 hover:text-black transition-all tracking-wider text-lg"
            >
              [REFRESH]
            </button>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="border-2 border-green-400 p-4 bg-black/80">
            <div className="text-base mb-2 opacity-60">TOTAL PACKETS</div>
            <div className="text-5xl font-bold glow-text">◎ {packets.length}</div>
          </div>
          <div className="border-2 border-green-400 p-4 bg-black/80">
            <div className="text-base mb-2 opacity-60">NORMAL TRAFFIC</div>
            <div className="text-5xl font-bold glow-text">✓ {normalCount}</div>
          </div>
          <div className="border-2 border-red-500 p-4 bg-black/80">
            <div className="text-base mb-2 text-red-500">THREATS DETECTED</div>
            <div className="text-5xl font-bold text-red-500 glow-text-red">⚠ {attackCount}</div>
          </div>
        </div>

        {/* Packet Feed */}
        <div className="border-4 border-green-400 p-4 bg-black/80">
          <h2 className="text-3xl mb-4 tracking-wider">╔═══ LIVE PACKET FEED ═══╗</h2>
          <div className="space-y-1 max-h-[500px] overflow-y-auto custom-scrollbar">
            {packets.length > 0 ? (
              packets.map((packet, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.01 }}
                  className={`p-3 border font-mono ${
                    packet.is_attack 
                      ? 'border-red-500 bg-red-500/10' 
                      : 'border-green-400/30 bg-green-400/5'
                  }`}
                >
                  <div className="flex justify-between items-start text-base">
                    <div className="flex-1">
                      <span className={packet.is_attack ? 'text-red-500' : 'text-green-400'}>
                        {packet.is_attack ? '[!] ATTACK' : '[✓] NORMAL'}
                      </span>
                      {packet.attack_type && (
                        <span className="ml-2 text-amber-400">
                          [{packet.attack_type.toUpperCase()}]
                        </span>
                      )}
                      <span className="ml-4 opacity-60">
                        {packet.protocol.toUpperCase()}
                      </span>
                      <span className="ml-2 opacity-60">
                        {packet.size}B
                      </span>
                    </div>
                    <div className="text-xs opacity-60">
                      {new Date(packet.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                  <div className="text-sm opacity-80 mt-1">
                    {packet.src_ip} → {packet.dst_ip}
                  </div>
                </motion.div>
              ))
            ) : (
              <div className="text-center opacity-60 py-8 text-xl">
                NO PACKETS CAPTURED YET // START CAPTURE TO BEGIN MONITORING
              </div>
            )}
          </div>
        </div>
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

        .glow-text-red {
          text-shadow: 
            0 0 5px #ff0040,
            0 0 10px #ff0040;
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