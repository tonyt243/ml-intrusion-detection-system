from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Get database URL from environment, fallback to SQLite for local testing
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./detections.db")

# Fix Railway PostgreSQL URL format
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

print(f"Connecting to database...")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Detection(Base):
    """Detection record stored in database"""
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    is_attack = Column(Boolean, nullable=False)
    alert_level = Column(String, nullable=False)
    reason = Column(String, nullable=False)
    attack_type = Column(String, nullable=True)
    
    # Random Forest predictions
    rf_confidence = Column(Float, nullable=True)
    rf_attack_probability = Column(Float, nullable=True)
    
    # Isolation Forest predictions
    iso_anomaly_score = Column(Float, nullable=True)
    
    # Key packet features
    protocol_type = Column(String, nullable=True)
    service = Column(String, nullable=True)
    flag = Column(String, nullable=True)
    src_bytes = Column(Integer, nullable=True)
    dst_bytes = Column(Integer, nullable=True)

    all_features = Column(String, nullable=True)


# Create tables if they don't exist
print("Creating database tables...")
Base.metadata.create_all(bind=engine)
print("Database ready")


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()