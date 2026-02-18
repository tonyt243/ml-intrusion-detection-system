# src/data/download_dataset.py
import pandas as pd
import os

def download_nsl_kdd():
    """Download NSL-KDD dataset"""
    
    # Column names for NSL-KDD
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
        'label', 'difficulty'
    ]
    
    # URLs
    train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
    test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
    
    print("Downloading NSL-KDD dataset...")
    print("="*60)
    
    # Create output directory
    os.makedirs('../../data/raw', exist_ok=True)
    
    # Download training data
    print("\n📥 Downloading training data...")
    train_df = pd.read_csv(train_url, names=columns)
    train_df.to_csv('../../data/raw/KDDTrain+.csv', index=False)
    print(f"✅ Training data saved: {train_df.shape[0]} records, {train_df.shape[1]} features")
    
    # Download test data
    print("\n📥 Downloading test data...")
    test_df = pd.read_csv(test_url, names=columns)
    test_df.to_csv('../../data/raw/KDDTest+.csv', index=False)
    print(f"✅ Test data saved: {test_df.shape[0]} records, {test_df.shape[1]} features")
    
    # Display basic info
    print("\n" + "="*60)
    print("📊 DATASET OVERVIEW")
    print("="*60)
    print(f"\nTraining set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    print("\n🎯 Attack Type Distribution (Training Data):")
    attack_counts = train_df['label'].value_counts()
    for label, count in attack_counts.head(10).items():
        print(f"  {label:20s}: {count:,}")
    
    # Create binary classification
    train_df['is_attack'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    normal_count = (train_df['is_attack'] == 0).sum()
    attack_count = (train_df['is_attack'] == 1).sum()
    
    print(f"\n📈 Binary Classification:")
    print(f"  Normal traffic: {normal_count:,} ({normal_count/len(train_df)*100:.1f}%)")
    print(f"  Attack traffic: {attack_count:,} ({attack_count/len(train_df)*100:.1f}%)")
    
    print("\n📋 Sample Data (first 3 rows):")
    print(train_df[['duration', 'protocol_type', 'service', 'src_bytes', 'dst_bytes', 'label']].head(3))
    
    print("\n✅ Dataset download complete!")
    print(f"📁 Files saved to: data/raw/")
    
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = download_nsl_kdd()