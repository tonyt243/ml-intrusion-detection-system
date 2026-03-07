import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

def preprocess_data(df, label_encoders=None, scaler=None, is_training=True):
    """Preprocess data inline"""
    df = df.copy()
    
    # Binary labels
    df['is_attack'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # Categorical encoding
    categorical_cols = ['protocol_type', 'service', 'flag']
    
    if is_training:
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    else:
        for col in categorical_cols:
            le = label_encoders[col]
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
    
    # Select features
    feature_cols = [col for col in df.columns if col not in ['label', 'difficulty', 'is_attack']]
    X = df[feature_cols]
    y = df['is_attack']
    
    # Scale
    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, y, feature_cols, label_encoders, scaler


class IDSModelTrainer:
    def __init__(self):
        self.models = {}
        self.metrics = {}
        
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier"""
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print("="*60)
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        print(f"\nTraining on {X_train.shape[0]:,} samples...")
        print(f"Features: {X_train.shape[1]}")
        
        rf_model.fit(X_train, y_train)
        
        print("✅ Random Forest training complete!")
        
        self.models['random_forest'] = rf_model
        return rf_model
    
    def train_isolation_forest(self, X_train):
        """Train Isolation Forest"""
        print("\n" + "="*60)
        print("TRAINING ISOLATION FOREST (ANOMALY DETECTION)")
        print("="*60)
        
        iso_model = IsolationForest(
            contamination=0.1,
            max_samples=256,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        print(f"\nTraining on {X_train.shape[0]:,} samples...")
        
        iso_model.fit(X_train)
        
        print("✅ Isolation Forest training complete!")
        
        self.models['isolation_forest'] = iso_model
        return iso_model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        print("\n" + "="*60)
        print(f"EVALUATING {model_name.upper()}")
        print("="*60)
        
        # Predictions
        if model_name == 'isolation_forest':
            y_pred_raw = model.predict(X_test)
            y_pred = np.where(y_pred_raw == -1, 1, 0)
        else:
            y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n📊 Performance Metrics:")
        print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"   F1-Score:  {f1:.4f}")
        
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Normal', 'Attack'],
                                   digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        plt.title(f'{model_name.replace("_", " ").title()} - Confusion Matrix', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        os.makedirs('../../data/models/plots', exist_ok=True)
        plot_path = f'../../data/models/plots/{model_name}_confusion_matrix.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Confusion matrix saved to: {plot_path}")
        plt.close()
        
        tn, fp, fn, tp = cm.ravel()
        self.metrics[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
        
        return accuracy, y_pred
    
    def plot_feature_importance(self, model, feature_names, model_name):
        """Plot feature importance"""
        if model_name != 'random_forest':
            return
        
        print("\n📊 Analyzing feature importance...")
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        top_n = 20
        top_indices = indices[:top_n]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), top_importances, color='steelblue', edgecolor='black')
        plt.yticks(range(top_n), top_features)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_n} Most Important Features - Random Forest', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        plot_path = '../../data/models/plots/feature_importance.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✅ Feature importance plot saved to: {plot_path}")
        plt.close()
        
        print(f"\n🔝 Top 10 Most Important Features:")
        for i in range(min(10, top_n)):
            print(f"   {i+1}. {top_features[i]:30s}: {top_importances[i]:.4f}")
    
    def save_models(self, label_encoders, scaler, feature_cols, path='../../data/models/'):
        """Save models and preprocessor"""
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = os.path.join(path, f'{name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ Saved {name} to: {model_path}")
        
        # Save preprocessor
        preprocessor_path = os.path.join(path, 'preprocessor.pkl')
        with open(preprocessor_path, 'wb') as f:
            pickle.dump({
                'label_encoders': label_encoders,
                'scaler': scaler,
                'feature_cols': feature_cols
            }, f)
        print(f"✅ Saved preprocessor to: {preprocessor_path}")
        
        # Save metrics
        metrics_path = os.path.join(path, 'metrics.pkl')
        with open(metrics_path, 'wb') as f:
            pickle.dump(self.metrics, f)
        print(f"✅ Saved metrics to: {metrics_path}")
    
    def print_summary(self):
        """Print summary"""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        for model_name, metrics in self.metrics.items():
            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
            print(f"  Precision: {metrics['precision']*100:.2f}%")
            print(f"  Recall:    {metrics['recall']*100:.2f}%")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  True Negatives:  {metrics['tn']:,}")
            print(f"  False Positives: {metrics['fp']:,}")
            print(f"  False Negatives: {metrics['fn']:,}")
            print(f"  True Positives:  {metrics['tp']:,}")


# Main
if __name__ == "__main__":
    print("="*60)
    print("ML-IDS MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("\n📥 Loading datasets...")
    train_df = pd.read_csv('../../data/raw/KDDTrain+.csv')
    test_df = pd.read_csv('../../data/raw/KDDTest+.csv')
    print(f"✅ Training data: {train_df.shape}")
    print(f"✅ Test data: {test_df.shape}")
    
    # Preprocess
    print("\n🔧 Preprocessing data...")
    X_train, y_train, feature_cols, label_encoders, scaler = preprocess_data(train_df, is_training=True)
    X_test, y_test, _, _, _ = preprocess_data(test_df, label_encoders, scaler, is_training=False)
    print(f"✅ Preprocessed {X_train.shape[0]:,} training samples")
    print(f"✅ Preprocessed {X_test.shape[0]:,} test samples")
    
    # Train
    trainer = IDSModelTrainer()
    
    rf_model = trainer.train_random_forest(X_train, y_train)
    rf_accuracy, rf_pred = trainer.evaluate_model(rf_model, X_test, y_test, 'random_forest')
    trainer.plot_feature_importance(rf_model, feature_cols, 'random_forest')
    
    iso_model = trainer.train_isolation_forest(X_train)
    iso_accuracy, iso_pred = trainer.evaluate_model(iso_model, X_test, y_test, 'isolation_forest')
    
    # Save
    trainer.save_models(label_encoders, scaler, feature_cols)
    
    # Summary
    trainer.print_summary()
    
    print("\n" + "="*60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*60)