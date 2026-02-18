# ML-Based Intrusion Detection System

Real-time network intrusion detection system using machine learning to identify and classify cyber attacks.

##  Project Overview

An intelligent Intrusion Detection System (IDS) that uses machine learning to detect network attacks in real-time with a professional web dashboard for monitoring.

##  Tech Stack

- **ML & Detection**: Python, Scapy, scikit-learn, Pandas, NumPy
- **Backend**: FastAPI, PostgreSQL, WebSockets
- **Frontend**: Next.js, TypeScript, Tailwind CSS, Recharts
- **Dataset**: NSL-KDD (125,973 training samples, 22,544 test samples)

##  Getting Started

### Prerequisites
- Python 3.10+
- pip
- Git

### Installation

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ml-intrusion-detection-system.git
cd ml-intrusion-detection-system
```

2. Create virtual environment
```bash
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download dataset
```bash
cd src/data
python download_dataset.py
```

##  Project Structure
```
ml-ids-project/
├── data/
│   ├── raw/              # NSL-KDD dataset
│   ├── processed/        # Preprocessed data
│   └── models/           # Trained ML models
├── src/
│   ├── data/             # Dataset scripts
│   ├── preprocessing/    # Data preprocessing
│   ├── models/           # ML model training
│   ├── detection/        # Real-time detection
│   └── api/              # FastAPI backend
├── notebooks/            # Jupyter notebooks for exploration
└── requirements.txt      # Python dependencies
```

##  Dataset Information

**NSL-KDD Dataset**
- Training samples: 125,973
- Test samples: 22,544  
- Features: 41
- Attack types: 23
- Distribution: 53.5% normal, 46.5% attacks

