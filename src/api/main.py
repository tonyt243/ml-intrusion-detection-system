from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
import os
from datetime import datetime
import asyncio
import json
from dotenv import load_dotenv


load_dotenv()

# Configuration
PORT = int(os.getenv("PORT", 8000))
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

def classify_attack_type(packet_features: dict) -> str:
    """
    Classify attack type based on packet characteristics
    Uses heuristic rules based on NSL-KDD attack patterns
    """
    # Extract key features
    serror_rate = packet_features.get('serror_rate', 0)
    count = packet_features.get('count', 0)
    dst_host_count = packet_features.get('dst_host_count', 0)
    flag = packet_features.get('flag', '')
    service = packet_features.get('service', '')
    protocol = packet_features.get('protocol_type', '')
    src_bytes = packet_features.get('src_bytes', 0)
    dst_bytes = packet_features.get('dst_bytes', 0)
    num_failed_logins = packet_features.get('num_failed_logins', 0)
    
    # Port Scan characteristics
    if serror_rate > 0.8 and count > 150 and flag in ['REJ', 'RSTO', 'S0']:
        return 'Port Scan'
    
    # DoS/DDoS characteristics  
    if count > 300 and serror_rate > 0.7 and service in ['http', 'private', 'ecr_i']:
        return 'DoS Attack'
    
    # Brute Force characteristics
    if num_failed_logins > 2 or (count > 50 and service in ['ftp', 'ssh', 'telnet']):
        return 'Brute Force'
    
    # IP Sweep/Probe characteristics
    if protocol == 'icmp' and dst_host_count > 100:
        return 'IP Sweep'
    
    if dst_host_count > 200 and count > 200:
        return 'Network Probe'
    
    # Neptune (SYN Flood)
    if flag == 'S0' and count > 300:
        return 'SYN Flood'
    
    # Smurf (ICMP Flood)
    if protocol == 'icmp' and count > 300:
        return 'ICMP Flood'
    
    # Data Exfiltration
    if src_bytes > 50000 or dst_bytes > 50000:
        return 'Data Exfiltration'
    
    # Generic attack if we can't classify
    return 'Unknown Attack'

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import RealTimeDetector

# Initialize FastAPI app
app = FastAPI(
    title="ML-IDS API",
    description="Real-time Intrusion Detection System using Machine Learning",
    version="1.0.0"
)

# CORS middleware (allows frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL,"http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector globally
print("\n Starting ML-IDS API...")
detector = RealTimeDetector(model_type='both')
print(" API ready to accept requests!\n")


# REQUEST/RESPONSE MODELS
class PacketFeatures(BaseModel):
    """Input model for packet features"""
    duration: int = 0
    protocol_type: str
    service: str
    flag: str
    src_bytes: int
    dst_bytes: int
    land: int = 0
    wrong_fragment: int = 0
    urgent: int = 0
    hot: int = 0
    num_failed_logins: int = 0
    logged_in: int = 0
    num_compromised: int = 0
    root_shell: int = 0
    su_attempted: int = 0
    num_root: int = 0
    num_file_creations: int = 0
    num_shells: int = 0
    num_access_files: int = 0
    num_outbound_cmds: int = 0
    is_host_login: int = 0
    is_guest_login: int = 0
    count: int
    srv_count: int
    serror_rate: float
    srv_serror_rate: float
    rerror_rate: float
    srv_rerror_rate: float
    same_srv_rate: float
    diff_srv_rate: float
    srv_diff_host_rate: float
    dst_host_count: int
    dst_host_srv_count: int
    dst_host_same_srv_rate: float
    dst_host_diff_srv_rate: float
    dst_host_same_src_port_rate: float
    dst_host_srv_diff_host_rate: float
    dst_host_serror_rate: float
    dst_host_srv_serror_rate: float
    dst_host_rerror_rate: float
    dst_host_srv_rerror_rate: float

class DetectionResponse(BaseModel):
    """Response model for detection results"""
    timestamp: str
    is_attack: bool
    alert_level: str
    reason: str
    attack_type: Optional[str] = None  
    predictions: Dict[str, Any]

class Statistics(BaseModel):
    """Response model for statistics"""
    total_packets: int
    attacks_detected: int
    normal_packets: int
    attack_rate: float


# API ENDPOINTS
@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "message": "ML-IDS API is running",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "detect": "/detect (POST)",
            "statistics": "/statistics (GET)",
            "health": "/health (GET)",
            "clear": "/clear (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "detector_loaded": detector is not None,
        "models": ["random_forest", "isolation_forest"]
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_packet(packet: PacketFeatures):
    """
    Detect if a packet is malicious
    
    Args:
        packet: Packet features following NSL-KDD format
        
    Returns:
        Detection result with predictions from both models
    """
    try:
        # Convert Pydantic model to dict
        packet_dict = packet.dict()
        
        # Run detection
        result = detector.detect(packet_dict)
        
        # Classify attack type if it's an attack
        attack_type = None
        if result['is_attack']:
            attack_type = classify_attack_type(packet_dict)
        
        # Format response
        response = {
            "timestamp": result['timestamp'].isoformat(),
            "is_attack": result['is_attack'],
            "alert_level": result.get('alert_level', 'UNKNOWN'),
            "reason": result.get('reason', 'No reason provided'),
            "attack_type": attack_type,  # NEW!
            "predictions": result['predictions']
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

@app.get("/statistics", response_model=Statistics)
async def get_statistics():
    """
    Get detection statistics
    
    Returns:
        Statistics including total packets, attacks detected, and attack rate
    """
    try:
        stats = detector.get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics error: {str(e)}")

@app.post("/clear")
async def clear_history():
    """
    Clear detection history
    
    Returns:
        Confirmation message
    """
    try:
        detector.clear_history()
        return {"message": "Detection history cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear error: {str(e)}")

@app.get("/recent")
async def get_recent_detections(limit: int = 10):
    """
    Get recent detections
    
    Args:
        limit: Number of recent detections to return (default 10)
        
    Returns:
        List of recent detection results
    """
    try:
        stats = detector.get_statistics()
        recent = stats.get('recent_detections', [])
        
        # Format for JSON serialization and ADD attack_type
        formatted = []
        for detection in recent[-limit:]:
            # Classify attack type if it's an attack
            attack_type = None
            if detection.get('is_attack'):
                attack_type = classify_attack_type(detection.get('packet_features', {}))
            
            formatted.append({
                "timestamp": detection['timestamp'].isoformat(),
                "is_attack": detection['is_attack'],
                "alert_level": detection.get('alert_level', 'UNKNOWN'),
                "reason": detection.get('reason', 'No reason'),
                "attack_type": attack_type,  # ADD THIS!
                "predictions": detection['predictions']
            })
        
        return {"recent_detections": formatted, "count": len(formatted)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recent detections error: {str(e)}")


# WEBSOCKET FOR REAL-TIME UPDATES
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f" WebSocket client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f" WebSocket client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time detection updates
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Send statistics every 2 seconds
            stats = detector.get_statistics()
            
            # Format timestamp for JSON
            recent = stats.get('recent_detections', [])
            formatted_recent = []
            for detection in recent[-5:]:  # Last 5
                formatted_recent.append({
                    "timestamp": detection['timestamp'].isoformat(),
                    "is_attack": detection['is_attack'],
                    "alert_level": detection.get('alert_level', 'UNKNOWN')
                })
            
            message = {
                "type": "statistics",
                "data": {
                    "total_packets": stats['total_packets'],
                    "attacks_detected": stats['attacks_detected'],
                    "normal_packets": stats['normal_packets'],
                    "attack_rate": stats['attack_rate'],
                    "recent_detections": formatted_recent
                }
            }
            
            await websocket.send_json(message)
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# STARTUP EVENT
@app.on_event("startup")
async def startup_event():
    """Run on API startup"""
    print("\n" + "="*60)
    print("ML-IDS API STARTED")
    print("="*60)
    print(" Swagger Docs: http://localhost:8000/docs")
    print(" WebSocket: ws://localhost:8000/ws")
    print(" API Root: http://localhost:8000")
    print("="*60 + "\n")


# RUN SERVER
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )