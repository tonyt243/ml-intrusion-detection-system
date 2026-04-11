from database import SessionLocal, Detection
from datetime import datetime

print("Testing database connection...")

# Create a session
db = SessionLocal()

# Create a test detection
test_detection = Detection(
    timestamp=datetime.utcnow(),
    is_attack=True,
    alert_level="HIGH",
    reason="Test detection",
    attack_type="Port Scan",
    rf_confidence=0.85,
    rf_attack_probability=0.85,
    iso_anomaly_score=-0.5,
    protocol_type="tcp",
    service="http",
    flag="SF",
    src_bytes=1000,
    dst_bytes=2000
)

# Add to database
db.add(test_detection)
db.commit()
db.refresh(test_detection)

print(f"Created detection with ID: {test_detection.id}")

# Query all detections
all_detections = db.query(Detection).all()
print(f"Total detections in database: {len(all_detections)}")

# Show the test detection
print(f"Test detection: Attack={test_detection.is_attack}, Type={test_detection.attack_type}")

db.close()
print("Database test complete!")