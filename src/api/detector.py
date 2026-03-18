import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os

#Initialize real-time detector with trained models
class RealTimeDetector:
    def __init__(self, model_path='../../data/models/', model_type='random_forest'):
        self.model_path = model_path
        self.model_type = model_type
        self.detection_history = []
        
        print("="*60)
        print("INITIALIZING REAL-TIME DETECTOR")
        print("="*60)
        
        # Load preprocessor
        print("\n Loading preprocessor...")
        preprocessor_file = os.path.join(model_path, 'preprocessor.pkl')
        with open(preprocessor_file, 'rb') as f:
            preprocessor_data = pickle.load(f)
            self.label_encoders = preprocessor_data['label_encoders']
            self.scaler = preprocessor_data['scaler']
            self.feature_cols = preprocessor_data['feature_cols']
        print(f" Preprocessor loaded with {len(self.feature_cols)} features")
        
        # Load Random Forest
        if model_type in ['random_forest', 'both']:
            print("\n Loading Random Forest model...")
            rf_file = os.path.join(model_path, 'random_forest.pkl')
            with open(rf_file, 'rb') as f:
                self.rf_model = pickle.load(f)
            print(f" Random Forest loaded ({len(self.rf_model.estimators_)} trees)")
        
        # Load Isolation Forest
        if model_type in ['isolation_forest', 'both']:
            print("\n Loading Isolation Forest model...")
            iso_file = os.path.join(model_path, 'isolation_forest.pkl')
            with open(iso_file, 'rb') as f:
                self.iso_model = pickle.load(f)
            print(f" Isolation Forest loaded")
        
        print("\n Detector ready for real-time predictions!")
        
    #Preprocess packet features for prediction
    def preprocess_packet(self, packet_features):

        # Convert to DataFrame
        df = pd.DataFrame([packet_features])
        
        # Encode categorical features
        for col in ['protocol_type', 'service', 'flag']:
            if col in df.columns:
                le = self.label_encoders.get(col)
                if le:
                    # Handle unseen categories
                    df[col] = df[col].apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])
        
        # Ensure we have all required features in correct order
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features
        
        # Select and order features
        X = df[self.feature_cols]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    #Detect if packet is malicious using trained models
    def detect(self, packet_features):
        # Preprocess
        X = self.preprocess_packet(packet_features)
        
        result = {
            'timestamp': datetime.now(),
            'packet_features': packet_features,
            'predictions': {}
        }
        
        # Random Forest prediction
        if hasattr(self, 'rf_model'):
            rf_pred = self.rf_model.predict(X)[0]
            rf_proba = self.rf_model.predict_proba(X)[0]
            
            result['predictions']['random_forest'] = {
                'is_attack': bool(rf_pred),
                'confidence': float(rf_proba[rf_pred]),
                'attack_probability': float(rf_proba[1]),
                'normal_probability': float(rf_proba[0])
            }
        
        # Isolation Forest prediction
        if hasattr(self, 'iso_model'):
            iso_pred = self.iso_model.predict(X)[0]
            iso_score = self.iso_model.score_samples(X)[0]
            
            # -1 = anomaly/attack, 1 = normal
            is_anomaly = (iso_pred == -1)
            
            result['predictions']['isolation_forest'] = {
                'is_attack': bool(is_anomaly),
                'anomaly_score': float(iso_score),
                'is_anomaly': bool(is_anomaly)
            }
        
        # Combined decision (if using both models)
        if self.model_type == 'both':
            rf_attack = result['predictions']['random_forest']['is_attack']
            iso_attack = result['predictions']['isolation_forest']['is_attack']
            
            # High alert if Random Forest says attack
            # Medium alert if only Isolation Forest says attack
            if rf_attack:
                result['alert_level'] = 'HIGH'
                result['is_attack'] = True
                result['reason'] = 'Known attack pattern detected'
            elif iso_attack:
                result['alert_level'] = 'MEDIUM'
                result['is_attack'] = True
                result['reason'] = 'Anomalous behavior detected'
            else:
                result['alert_level'] = 'NONE'
                result['is_attack'] = False
                result['reason'] = 'Normal traffic'
        else:
            # Single model decision
            if self.model_type == 'random_forest':
                result['is_attack'] = result['predictions']['random_forest']['is_attack']
            else:
                result['is_attack'] = result['predictions']['isolation_forest']['is_attack']
        
        # Add to history
        self.detection_history.append(result)
        
        return result
    
    #Get detection statistics
    def get_statistics(self):
       
        if not self.detection_history:
            return {
                'total_packets': 0,
                'attacks_detected': 0,
                'normal_packets': 0,
                'attack_rate': 0.0
            }
        
        total = len(self.detection_history)
        attacks = sum(1 for d in self.detection_history if d.get('is_attack', False))
        
        return {
            'total_packets': total,
            'attacks_detected': attacks,
            'normal_packets': total - attacks,
            'attack_rate': attacks / total if total > 0 else 0.0,
            'recent_detections': self.detection_history[-10:]
        }
    
    #Clear detection history
    def clear_history(self):
        self.detection_history = []
        print(" Detection history cleared")


# Test the detector
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING REAL-TIME DETECTOR")
    print("="*60)
    
    # Initialize detector with both models
    detector = RealTimeDetector(model_type='both')
    
    # Test packet 1: Normal-looking traffic
    print("\n" + "-"*60)
    print("TEST 1: Normal HTTP Request")
    print("-"*60)
    
    normal_packet = {
        'duration': 0,
        'protocol_type': 'tcp',
        'service': 'http',
        'flag': 'SF',
        'src_bytes': 500,
        'dst_bytes': 1500,
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': 0,
        'num_failed_logins': 0,
        'logged_in': 1,
        'num_compromised': 0,
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,
        'num_file_creations': 0,
        'num_shells': 0,
        'num_access_files': 0,
        'num_outbound_cmds': 0,
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': 5,
        'srv_count': 5,
        'serror_rate': 0.0,
        'srv_serror_rate': 0.0,
        'rerror_rate': 0.0,
        'srv_rerror_rate': 0.0,
        'same_srv_rate': 1.0,
        'diff_srv_rate': 0.0,
        'srv_diff_host_rate': 0.0,
        'dst_host_count': 10,
        'dst_host_srv_count': 10,
        'dst_host_same_srv_rate': 1.0,
        'dst_host_diff_srv_rate': 0.0,
        'dst_host_same_src_port_rate': 0.1,
        'dst_host_srv_diff_host_rate': 0.0,
        'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0,
        'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0
    }
    
    result1 = detector.detect(normal_packet)
    
    if result1['is_attack']:
        print(f" ATTACK DETECTED!")
        print(f"   Alert Level: {result1.get('alert_level', 'N/A')}")
        print(f"   Reason: {result1.get('reason', 'N/A')}")
    else:
        print(f" NORMAL TRAFFIC")
    
    if 'random_forest' in result1['predictions']:
        rf = result1['predictions']['random_forest']
        print(f"\n   Random Forest:")
        print(f"      Prediction: {'Attack' if rf['is_attack'] else 'Normal'}")
        print(f"      Confidence: {rf['confidence']*100:.2f}%")
        print(f"      Attack Probability: {rf['attack_probability']*100:.2f}%")
    
    if 'isolation_forest' in result1['predictions']:
        iso = result1['predictions']['isolation_forest']
        print(f"\n   Isolation Forest:")
        print(f"      Prediction: {'Anomaly' if iso['is_attack'] else 'Normal'}")
        print(f"      Anomaly Score: {iso['anomaly_score']:.4f}")
    
    # Test packet 2: Suspicious port scan
    print("\n" + "-"*60)
    print("TEST 2: Suspicious Port Scan")
    print("-"*60)
    
    attack_packet = {
        'duration': 0,
        'protocol_type': 'tcp',
        'service': 'private',
        'flag': 'REJ',
        'src_bytes': 0,
        'dst_bytes': 0,
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': 0,
        'num_failed_logins': 0,
        'logged_in': 0,
        'num_compromised': 0,
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,
        'num_file_creations': 0,
        'num_shells': 0,
        'num_access_files': 0,
        'num_outbound_cmds': 0,
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': 500,
        'srv_count': 500,
        'serror_rate': 1.0,
        'srv_serror_rate': 1.0,
        'rerror_rate': 0.0,
        'srv_rerror_rate': 0.0,
        'same_srv_rate': 1.0,
        'diff_srv_rate': 0.0,
        'srv_diff_host_rate': 0.0,
        'dst_host_count': 255,
        'dst_host_srv_count': 255,
        'dst_host_same_srv_rate': 1.0,
        'dst_host_diff_srv_rate': 0.0,
        'dst_host_same_src_port_rate': 0.0,
        'dst_host_srv_diff_host_rate': 0.0,
        'dst_host_serror_rate': 1.0,
        'dst_host_srv_serror_rate': 1.0,
        'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0
    }
    
    result2 = detector.detect(attack_packet)
    
    if result2['is_attack']:
        print(f" ATTACK DETECTED!")
        print(f"   Alert Level: {result2.get('alert_level', 'N/A')}")
        print(f"   Reason: {result2.get('reason', 'N/A')}")
    else:
        print(f" NORMAL TRAFFIC")
    
    if 'random_forest' in result2['predictions']:
        rf = result2['predictions']['random_forest']
        print(f"\n   Random Forest:")
        print(f"      Prediction: {'Attack' if rf['is_attack'] else 'Normal'}")
        print(f"      Confidence: {rf['confidence']*100:.2f}%")
        print(f"      Attack Probability: {rf['attack_probability']*100:.2f}%")
    
    if 'isolation_forest' in result2['predictions']:
        iso = result2['predictions']['isolation_forest']
        print(f"\n   Isolation Forest:")
        print(f"      Prediction: {'Anomaly' if iso['is_attack'] else 'Normal'}")
        print(f"      Anomaly Score: {iso['anomaly_score']:.4f}")
    
    # Show statistics
    print("\n" + "="*60)
    print("DETECTION STATISTICS")
    print("="*60)
    stats = detector.get_statistics()
    print(f"Total packets analyzed: {stats['total_packets']}")
    print(f"Attacks detected: {stats['attacks_detected']}")
    print(f"Normal packets: {stats['normal_packets']}")
    print(f"Attack rate: {stats['attack_rate']*100:.2f}%")
    print("\n Detector test complete!")