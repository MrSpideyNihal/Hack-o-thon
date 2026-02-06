import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_baseline_data(n_samples=30000):
    """
    Generate HEALTHY baseline data (model was trained on this)
    This represents normal conditions
    """
    np.random.seed(42)
    
    data = {
        'timestamp': [datetime.now() - timedelta(days=np.random.randint(30, 90)) 
                     for _ in range(n_samples)],
        'age': np.random.normal(45, 12, n_samples).clip(18, 80),
        'income': np.random.normal(50000, 15000, n_samples).clip(20000, 200000),
        'credit_score': np.random.normal(680, 50, n_samples).clip(300, 850),
        'debt_ratio': np.random.beta(2, 5, n_samples) * 0.8,
        'employment_years': np.random.exponential(5, n_samples).clip(0, 40),
        'zip_code': np.random.choice(range(90001, 90100), n_samples),
        'loan_amount': np.random.normal(150000, 50000, n_samples).clip(50000, 500000),
    }
    
    df = pd.DataFrame(data)
    
    # Model decision logic (simplified)
    df['score'] = (
        (df['credit_score'] - 600) * 0.4 +
        (df['income'] / 1000) * 0.3 -
        (df['debt_ratio'] * 200) +
        (df['employment_years'] * 2)
    )
    df['prediction'] = (df['score'] > 100).astype(int)
    df['confidence'] = 1 / (1 + np.exp(-df['score'] / 50))  # Sigmoid
    
    return df

def generate_drift_scenario_A(n_samples=5000):
    """
    SCENARIO A: Economic Downturn
    - Income distribution shifts DOWN (people earning less)
    - This is BAD - model trained on higher incomes, now seeing lower
    """
    np.random.seed(100)
    
    data = {
        'timestamp': [datetime.now() - timedelta(hours=np.random.randint(0, 48)) 
                     for _ in range(n_samples)],
        'age': np.random.normal(45, 12, n_samples).clip(18, 80),
        'income': np.random.normal(40000, 15000, n_samples).clip(20000, 200000),  # SHIFTED DOWN
        'credit_score': np.random.normal(680, 50, n_samples).clip(300, 850),
        'debt_ratio': np.random.beta(2, 5, n_samples) * 0.8,
        'employment_years': np.random.exponential(5, n_samples).clip(0, 40),
        'zip_code': np.random.choice(range(90001, 90100), n_samples),
        'loan_amount': np.random.normal(150000, 50000, n_samples).clip(50000, 500000),
    }
    
    df = pd.DataFrame(data)
    df['score'] = (
        (df['credit_score'] - 600) * 0.4 +
        (df['income'] / 1000) * 0.3 -
        (df['debt_ratio'] * 200) +
        (df['employment_years'] * 2)
    )
    df['prediction'] = (df['score'] > 100).astype(int)
    df['confidence'] = 1 / (1 + np.exp(-df['score'] / 50))
    
    return df

def generate_drift_scenario_B(n_samples=5000):
    """
    SCENARIO B: Credit Score Degradation
    - Credit scores dropping across population
    """
    np.random.seed(200)
    
    data = {
        'timestamp': [datetime.now() - timedelta(hours=np.random.randint(0, 48)) 
                     for _ in range(n_samples)],
        'age': np.random.normal(45, 12, n_samples).clip(18, 80),
        'income': np.random.normal(50000, 15000, n_samples).clip(20000, 200000),
        'credit_score': np.random.normal(650, 50, n_samples).clip(300, 850),  # SHIFTED DOWN
        'debt_ratio': np.random.beta(2, 5, n_samples) * 0.8,
        'employment_years': np.random.exponential(5, n_samples).clip(0, 40),
        'zip_code': np.random.choice(range(90001, 90100), n_samples),
        'loan_amount': np.random.normal(150000, 50000, n_samples).clip(50000, 500000),
    }
    
    df = pd.DataFrame(data)
    df['score'] = (
        (df['credit_score'] - 600) * 0.4 +
        (df['income'] / 1000) * 0.3 -
        (df['debt_ratio'] * 200) +
        (df['employment_years'] * 2)
    )
    df['prediction'] = (df['score'] > 100).astype(int)
    df['confidence'] = 1 / (1 + np.exp(-df['score'] / 50))
    
    return df

def generate_drift_scenario_C(n_samples=5000):
    """
    SCENARIO C: Regional Economic Crisis
    - Specific zip codes (90001-90010) see income crash
    - Subpopulation drift - affects SPECIFIC GROUP only
    """
    np.random.seed(300)
    
    data = {
        'timestamp': [datetime.now() - timedelta(hours=np.random.randint(0, 48)) 
                     for _ in range(n_samples)],
        'age': np.random.normal(45, 12, n_samples).clip(18, 80),
        'income': np.random.normal(50000, 15000, n_samples).clip(20000, 200000),
        'credit_score': np.random.normal(680, 50, n_samples).clip(300, 850),
        'debt_ratio': np.random.beta(2, 5, n_samples) * 0.8,
        'employment_years': np.random.exponential(5, n_samples).clip(0, 40),
        'zip_code': np.random.choice(range(90001, 90100), n_samples),
        'loan_amount': np.random.normal(150000, 50000, n_samples).clip(50000, 500000),
    }
    
    df = pd.DataFrame(data)
    
    # REGIONAL CRISIS: Zip codes 90001-90010 see income drop
    crisis_zips = (df['zip_code'] >= 90001) & (df['zip_code'] <= 90010)
    df.loc[crisis_zips, 'income'] = df.loc[crisis_zips, 'income'] * 0.6  # 40% income drop
    
    df['score'] = (
        (df['credit_score'] - 600) * 0.4 +
        (df['income'] / 1000) * 0.3 -
        (df['debt_ratio'] * 200) +
        (df['employment_years'] * 2)
    )
    df['prediction'] = (df['score'] > 100).astype(int)
    df['confidence'] = 1 / (1 + np.exp(-df['score'] / 50))
    
    return df

if __name__ == "__main__":
    print("Generating datasets...")
    
    baseline = generate_baseline_data(30000)
    baseline.to_csv('baseline_data.csv', index=False)
    print(f"[OK] Baseline data: {len(baseline)} samples")
    
    drift_a = generate_drift_scenario_A(5000)
    drift_a.to_csv('drift_scenario_A.csv', index=False)
    print(f"[OK] Drift Scenario A (Income shift): {len(drift_a)} samples")
    
    drift_b = generate_drift_scenario_B(5000)
    drift_b.to_csv('drift_scenario_B.csv', index=False)
    print(f"[OK] Drift Scenario B (Credit degradation): {len(drift_b)} samples")
    
    drift_c = generate_drift_scenario_C(5000)
    drift_c.to_csv('drift_scenario_C.csv', index=False)
    print(f"[OK] Drift Scenario C (Regional crisis): {len(drift_c)} samples")
    
    print("\nDataset generation complete!")
