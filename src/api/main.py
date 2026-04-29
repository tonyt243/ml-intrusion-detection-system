from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
import os
from datetime import datetime, timedelta
import asyncio
import json
from dotenv import load_dotenv
from database import get_db, Detection as DBDetection
from sqlalchemy import func
from sqlalchemy.orm import Session
from fastapi import Depends
from packet_capture import PacketCapturer


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

# Initialize packet capturer
capturer = PacketCapturer()
captured_packets = []

def handle_captured_packet(packet_data):
    """Callback for captured packets"""
    # Run detection on captured packet
    try:
        features = packet_data['features']
        result = detector.detect(features)
        
        # Add to captured packets list
        captured_packets.append({
            'timestamp': packet_data['timestamp'],
            'src_ip': packet_data['src_ip'],
            'dst_ip': packet_data['dst_ip'],
            'protocol': packet_data['protocol'],
            'size': packet_data['size'],
            'is_attack': result['is_attack'],
            'alert_level': result.get('alert_level', 'UNKNOWN'),
            'attack_type': classify_attack_type(features) if result['is_attack'] else None
        })
        
        # Keep only last 100 packets
        if len(captured_packets) > 100:
            captured_packets.pop(0)
            
    except Exception as e:
        print(f"Error processing captured packet: {e}")

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
async def detect_packet(packet: PacketFeatures, db: Session = Depends(get_db)):
    """
    Detect if a packet is malicious and save to database
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
        
        # Save to database
        db_detection = DBDetection(
            timestamp=result['timestamp'],
            is_attack=result['is_attack'],
            alert_level=result.get('alert_level', 'UNKNOWN'),
            reason=result.get('reason', 'No reason provided'),
            attack_type=attack_type,
            rf_confidence=result['predictions'].get('random_forest', {}).get('confidence'),
            rf_attack_probability=result['predictions'].get('random_forest', {}).get('attack_probability'),
            iso_anomaly_score=result['predictions'].get('isolation_forest', {}).get('anomaly_score'),
            protocol_type=packet_dict.get('protocol_type'),
            service=packet_dict.get('service'),
            flag=packet_dict.get('flag'),
            src_bytes=packet_dict.get('src_bytes'),
            dst_bytes=packet_dict.get('dst_bytes'),
            all_features=json.dumps(packet_dict) 
        )
        
        db.add(db_detection)
        db.commit()
        db.refresh(db_detection)
        
        # Format response
        response = {
            "timestamp": result['timestamp'].isoformat(),
            "is_attack": result['is_attack'],
            "alert_level": result.get('alert_level', 'UNKNOWN'),
            "reason": result.get('reason', 'No reason provided'),
            "attack_type": attack_type,
            "predictions": result['predictions']
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

@app.get("/statistics", response_model=Statistics)
async def get_statistics(db: Session = Depends(get_db)):
    """
    Get detection statistics from database
    """
    try:
        total_packets = db.query(DBDetection).count()
        attacks_detected = db.query(DBDetection).filter(DBDetection.is_attack == True).count()
        normal_packets = total_packets - attacks_detected
        attack_rate = attacks_detected / total_packets if total_packets > 0 else 0.0
        
        return {
            "total_packets": total_packets,
            "attacks_detected": attacks_detected,
            "normal_packets": normal_packets,
            "attack_rate": attack_rate
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics error: {str(e)}")

@app.post("/clear")
async def clear_history(db: Session = Depends(get_db)):
    """
    Clear all detections from database
    """
    try:
        db.query(DBDetection).delete()
        db.commit()
        detector.clear_history()  # Also clear in-memory
        return {"message": "Detection history cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear error: {str(e)}")

@app.get("/recent")
async def get_recent_detections(limit: int = 10, db: Session = Depends(get_db)):
    """
    Get recent detections from database
    """
    try:
        # Query recent detections, ordered by timestamp descending
        recent = db.query(DBDetection).order_by(DBDetection.timestamp.desc()).limit(limit).all()
        
        # Format for response
        formatted = []
        for detection in recent:
            formatted.append({
                "id": detection.id,
                "timestamp": detection.timestamp.isoformat(),
                "is_attack": detection.is_attack,
                "alert_level": detection.alert_level,
                "reason": detection.reason,
                "attack_type": detection.attack_type,
                "predictions": {
                    "random_forest": {
                        "is_attack": detection.is_attack,
                        "confidence": detection.rf_confidence or 0.0,
                        "attack_probability": detection.rf_attack_probability or 0.0
                    },
                    "isolation_forest": {
                        "is_attack": detection.is_attack,
                        "anomaly_score": detection.iso_anomaly_score or 0.0
                    }
                }
            })
        
        return {"recent_detections": formatted, "count": len(formatted)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recent detections error: {str(e)}")

@app.get("/analytics/hourly")
async def get_hourly_stats(hours: int = 168, db: Session = Depends(get_db)):
    """
    Get detection statistics grouped by hour for the last N hours
    """
    try:
        # Calculate time threshold
        time_threshold = datetime.utcnow() - timedelta(hours=hours)
        
        # Query detections in time range
        detections = db.query(DBDetection).filter(
            DBDetection.timestamp >= time_threshold
        ).all()
        
        # Group by hour
        hourly_data = {}
        for detection in detections:
            hour_key = detection.timestamp.strftime('%Y-%m-%d')
            if hour_key not in hourly_data:
                hourly_data[hour_key] = {'total': 0, 'attacks': 0, 'normal': 0}
            
            hourly_data[hour_key]['total'] += 1
            if detection.is_attack:
                hourly_data[hour_key]['attacks'] += 1
            else:
                hourly_data[hour_key]['normal'] += 1
        
        # Format for chart
        result = []
        for hour, data in sorted(hourly_data.items()):
            result.append({
                'date': hour,
                'total': data['total'],
                'attacks': data['attacks'],
                'normal': data['normal'],
                'attack_rate': data['attacks'] / data['total'] if data['total'] > 0 else 0
            })
        
        return {"data": result, "hours": hours}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")


@app.get("/analytics/attack-types")
async def get_attack_type_breakdown(db: Session = Depends(get_db)):
    """
    Get breakdown of attacks by type
    """
    try:
        # Query all attacks
        attacks = db.query(DBDetection).filter(
            DBDetection.is_attack == True,
            DBDetection.attack_type.isnot(None)
        ).all()
        
        # Count by type
        type_counts = {}
        for attack in attacks:
            attack_type = attack.attack_type or 'Unknown'
            type_counts[attack_type] = type_counts.get(attack_type, 0) + 1
        
        # Format for chart
        result = [
            {'type': attack_type, 'count': count}
            for attack_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return {"data": result, "total_attacks": len(attacks)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attack type breakdown error: {str(e)}")


@app.get("/analytics/timeline")
async def get_detection_timeline(limit: int = 100, db: Session = Depends(get_db)):
    """
    Get timeline of detections for visualization
    """
    try:
        # Get recent detections
        detections = db.query(DBDetection).order_by(
            DBDetection.timestamp.desc()
        ).limit(limit).all()
        
        # Format for time series chart
        result = []
        for detection in reversed(detections):  # Reverse to get chronological order
            result.append({
                'timestamp': detection.timestamp.isoformat(),
                'is_attack': detection.is_attack,
                'attack_type': detection.attack_type,
                'confidence': detection.rf_confidence
            })
        
        return {"data": result, "count": len(result)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Timeline error: {str(e)}")

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

@app.post("/capture/start")
async def start_capture(interface: Optional[str] = None):
    """Start live packet capture"""
    try:
        success = capturer.start_capture(interface=interface, callback=handle_captured_packet)
        if success:
            return {"message": "Packet capture started", "interface": interface or "default"}
        else:
            return {"message": "Capture already running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Capture start error: {str(e)}")

@app.post("/capture/stop")
async def stop_capture():
    """Stop live packet capture"""
    try:
        capturer.stop_capture()
        return {"message": "Packet capture stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Capture stop error: {str(e)}")

@app.get("/capture/status")
async def get_capture_status():
    """Get capture status"""
    status = capturer.get_status()
    return {
        "is_capturing": status['is_capturing'],
        "packets_captured": len(captured_packets)
    }

@app.get("/capture/packets")
async def get_captured_packets(limit: int = 100):
    """Get recently captured packets"""
    return {
        "packets": captured_packets[-limit:],
        "count": len(captured_packets)
    }

@app.get("/detection/{detection_id}")
async def get_detection_detail(detection_id: int, db: Session = Depends(get_db)):
    """
    Get detailed information about a specific detection
    """
    try:
        detection = db.query(DBDetection).filter(DBDetection.id == detection_id).first()
        
        if not detection:
            raise HTTPException(status_code=404, detail="Detection not found")
        
        # Parse all features from JSON
        all_features = {}
        if detection.all_features:
            try:
                all_features = json.loads(detection.all_features)
            except:
                all_features = {}
        
        # Extract key features for display
        key_features = {
            # Basic connection info
            "duration": all_features.get("duration", 0),
            "protocol_type": all_features.get("protocol_type", "N/A"),
            "service": all_features.get("service", "N/A"),
            "flag": all_features.get("flag", "N/A"),
            
            # Byte counts
            "src_bytes": all_features.get("src_bytes", 0),
            "dst_bytes": all_features.get("dst_bytes", 0),
            
            # Connection counts
            "count": all_features.get("count", 0),
            "srv_count": all_features.get("srv_count", 0),
            "dst_host_count": all_features.get("dst_host_count", 0),
            
            # Error rates
            "serror_rate": all_features.get("serror_rate", 0.0),
            "srv_serror_rate": all_features.get("srv_serror_rate", 0.0),
            "dst_host_serror_rate": all_features.get("dst_host_serror_rate", 0.0),
            
            # Important rates
            "same_srv_rate": all_features.get("same_srv_rate", 0.0),
            "dst_host_same_srv_rate": all_features.get("dst_host_same_srv_rate", 0.0),
            
            # Security indicators
            "logged_in": all_features.get("logged_in", 0),
            "num_failed_logins": all_features.get("num_failed_logins", 0),
            "root_shell": all_features.get("root_shell", 0),
            "num_compromised": all_features.get("num_compromised", 0)
        }
        
        # Format detailed response
        detail = {
            "id": detection.id,
            "timestamp": detection.timestamp.isoformat(),
            "is_attack": detection.is_attack,
            "alert_level": detection.alert_level,
            "reason": detection.reason,
            "attack_type": detection.attack_type,
            
            # Model predictions
            "predictions": {
                "random_forest": {
                    "confidence": detection.rf_confidence,
                    "attack_probability": detection.rf_attack_probability,
                    "verdict": "ATTACK" if detection.rf_attack_probability and detection.rf_attack_probability > 0.5 else "NORMAL"
                },
                "isolation_forest": {
                    "anomaly_score": detection.iso_anomaly_score,
                    "verdict": "ANOMALY" if detection.iso_anomaly_score and detection.iso_anomaly_score < -0.5 else "NORMAL"
                }
            },
            
            # Key packet features
            "key_features": key_features
        }
        
        return detail
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving detection: {str(e)}")

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