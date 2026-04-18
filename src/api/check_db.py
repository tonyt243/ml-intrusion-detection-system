import os
from database import SessionLocal, Detection

db = SessionLocal()


print("DATABASE CHECK")


database_url = os.getenv("DATABASE_URL", "sqlite:///./detections.db")
if "postgres" in database_url:
    print("Connected to: PostgreSQL (Railway)")
else:
    print("Connected to: SQLite (Local)")

# Count total detections
total = db.query(Detection).count()
print(f"\nTotal detections in database: {total}")

# Count attacks vs normal
attacks = db.query(Detection).filter(Detection.is_attack == True).count()
normal = total - attacks
print(f"Attacks: {attacks}")
print(f"Normal: {normal}")


print("\nLast 10 detections:")
recent = db.query(Detection).order_by(Detection.timestamp.desc()).limit(10).all()

for d in recent:
    status = "ATTACK" if d.is_attack else "NORMAL"
    attack_type = f" ({d.attack_type})" if d.attack_type else ""
    confidence = f"{d.rf_confidence:.2f}" if d.rf_confidence else "N/A"
    print(f"  [{d.timestamp}] {status}{attack_type} - Confidence: {confidence}")

db.close()
