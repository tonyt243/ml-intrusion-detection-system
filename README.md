# ML-Based Intrusion Detection System

A real-time network intrusion detection system using machine learning algorithms to identify cyber threats. Built with Random Forest and Isolation Forest models achieving 76.77% accuracy on the NSL-KDD dataset.

## Live Demo

- **Dashboard**: [https://ml-intrusion-detection-system.vercel.app/](https://ml-intrusion-detection-system.vercel.app/)


## Overview

This system monitors network traffic in real-time and uses machine learning to detect malicious activity. It combines two complementary algorithms - Random Forest for classification and Isolation Forest for anomaly detection - to provide comprehensive threat identification with automatic attack type classification.

## Features

### Machine Learning Detection
- **Random Forest Classifier**: 76.77% accuracy, 96.73% precision, 61.27% recall
- **Isolation Forest**: Anomaly detection with 57.59% accuracy
- **Hybrid Detection**: Combines both models for enhanced accuracy
- **Real-time Classification**: Identifies attack types (Port Scan, DoS, Brute Force, IP Sweep)

### User Interface
- **Retro Terminal Design**: 1980s CRT monitor aesthetic with phosphor green glow and scanline effects
- **Real-time Dashboard**: Live detection feed with confidence scores and threat levels
- **Historical Analytics**: Time-series charts showing attack trends over 24 hours
- **Attack Type Breakdown**: Visual distribution of detected attack vectors
- **Detection Timeline**: Scrollable history of all network events

### Data Persistence
- PostgreSQL database stores all detections with full packet metadata
- Historical data survives server restarts
- Enables long-term trend analysis and reporting

## Architecture
```
User Interface (Vercel - Next.js)
|
| HTTP/REST API
|
Backend API (Railway - FastAPI)
|
|-- ML Detection Engine
|   |-- Random Forest Model
|   |-- Isolation Forest Model
|   |-- Attack Classifier
|
|-- PostgreSQL Database (Railway)
|-- Detection History
|-- Analytics Data
```

## Technology Stack

**Frontend**
- Next.js 15 with TypeScript
- Tailwind CSS for styling
- Framer Motion for animations
- Recharts for data visualization

**Backend**
- FastAPI (Python web framework)
- SQLAlchemy ORM
- Uvicorn ASGI server

**Machine Learning**
- Scikit-learn (Random Forest, Isolation Forest)
- Pandas for data processing
- NumPy for numerical operations

**Database & Deployment**
- PostgreSQL database
- Vercel (frontend hosting)
- Railway (backend + database hosting)

## Dataset

Trained on the NSL-KDD dataset, an improved version of the KDD'99 dataset widely used for network intrusion detection research.

**Dataset Statistics**
- Training samples: 125,973
- Test samples: 22,544
- Features: 41 network connection features
- Attack categories: DoS, Probe, R2L, U2R

**Key Features Used**
1. src_bytes (19.64% importance)
2. dst_bytes (10.44% importance)
3. same_srv_rate (8.49% importance)
4. Protocol type, service, flag
5. Connection counts and error rates

## Model Performance

### Random Forest Classifier
- Accuracy: 76.77%
- Precision: 96.73%
- Recall: 61.27%
- F1-Score: 75.07%
- Configuration: 100 trees, max depth 20

### Isolation Forest
- Accuracy: 57.59%
- Contamination: 0.1
- Purpose: Anomaly detection for unknown attacks

## Attack Classification

The system automatically classifies detected attacks into specific types based on packet characteristics:

- **Port Scan**: High error rates (>80%) with rejected connections, multiple destination hosts
- **DoS Attack**: Connection count >300, high error rates, targets specific services
- **Brute Force**: Multiple failed login attempts, targets authentication services (FTP, SSH)
- **IP Sweep**: ICMP protocol with high destination host count (>100)
- **SYN Flood**: TCP flag S0 with extremely high connection counts
- **ICMP Flood**: ICMP protocol with connection count >300
- **Data Exfiltration**: Unusually large data transfers (>50KB)

## API Endpoints

### Core Detection
- `POST /detect` - Analyze network packet and return threat assessment
- `GET /statistics` - Overall detection statistics (total packets, attacks, attack rate)
- `GET /recent` - Recent detections with configurable limit
- `POST /clear` - Clear detection history from database

### Analytics
- `GET /analytics/hourly` - Hourly detection counts for last 24 hours
- `GET /analytics/attack-types` - Distribution of attack types
- `GET /analytics/timeline` - Chronological detection timeline

### System
- `GET /health` - API health check
- `GET /docs` - Interactive Swagger documentation

## Project Structure
```
ml-intrusion-detection-system/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI application
│   │   ├── detector.py          # ML detection engine
│   │   ├── database.py          # SQLAlchemy models
│   │   └── models/              # Serialized ML models
│   ├── models/
│   │   └── train_models.py      # Model training pipeline
│   └── detection/
│       └── detector.py          # Detection logic
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # Main dashboard
│   │   └── analytics/
│   │       └── page.tsx         # Analytics page
│   └── package.json
├── data/
│   ├── raw/                     # NSL-KDD dataset
│   └── models/                  # Trained models
└── README.md
```

