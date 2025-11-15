"""
BiasGuard 2.0 - Prediction Test Data Generator
Generates prediction CSVs with known bias patterns for testing

No model training required! Just generates what a biased model would output.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

np.random.seed(42)
random.seed(42)

class BiasedPredictionGenerator:
    """Generate prediction logs with controllable bias levels"""
    
    def __init__(self, n_samples=5000):
        self.n_samples = n_samples
    
    def generate_predictions(self, bias_level="borderline"):
        """
        Generate predictions with specified bias level
        
        Args:
            bias_level: "none", "borderline", "moderate", "severe"
        
        Returns:
            DataFrame with predictions and sensitive attributes
        """
        
        print(f"\n{'='*70}")
        print(f"Generating {self.n_samples:,} predictions with {bias_level.upper()} bias")
        print("="*70)
        
        # Generate demographics
        races = np.random.choice(
            ['White', 'Black', 'Hispanic', 'Asian', 'Other'],
            size=self.n_samples,
            p=[0.60, 0.15, 0.18, 0.05, 0.02]
        )
        
        genders = np.random.choice(
            ['Male', 'Female', 'Non-Binary'],
            size=self.n_samples,
            p=[0.48, 0.50, 0.02]
        )
        
        ages = np.clip(np.random.normal(36, 12, self.n_samples).astype(int), 18, 75)
        
        marital_statuses = []
        for age in ages:
            if age < 25:
                status = np.random.choice(['Single', 'Married'], p=[0.85, 0.15])
            elif age < 35:
                status = np.random.choice(['Single', 'Married'], p=[0.55, 0.45])
            else:
                status = np.random.choice(['Single', 'Married', 'Divorced'], p=[0.25, 0.60, 0.15])
            marital_statuses.append(status)
        
        # Define bias parameters for each level
        bias_params = {
            "none": {
                "base_rate": 0.55,
                "race_penalties": {"Black": 0.00, "Hispanic": 0.00, "Asian": 0.00},
                "gender_penalties": {"Female": 0.00, "Non-Binary": 0.00},
                "age_penalties": {"young": 0.00, "old": 0.00}
            },
            "borderline": {
                "base_rate": 0.55,
                "race_penalties": {"Black": 0.12, "Hispanic": 0.08, "Asian": -0.03},
                "gender_penalties": {"Female": 0.06, "Non-Binary": 0.09},
                "age_penalties": {"young": 0.05, "old": 0.04}
            },
            "moderate": {
                "base_rate": 0.55,
                "race_penalties": {"Black": 0.22, "Hispanic": 0.15, "Asian": -0.05},
                "gender_penalties": {"Female": 0.12, "Non-Binary": 0.16},
                "age_penalties": {"young": 0.10, "old": 0.08}
            },
            "severe": {
                "base_rate": 0.55,
                "race_penalties": {"Black": 0.35, "Hispanic": 0.25, "Asian": -0.08},
                "gender_penalties": {"Female": 0.20, "Non-Binary": 0.25},
                "age_penalties": {"young": 0.15, "old": 0.12}
            }
        }
        
        params = bias_params[bias_level]
        
        # Generate predictions with bias
        predictions = []
        ground_truths = []  # Simulate actual outcomes
        
        for i in range(self.n_samples):
            # Start with base approval probability
            approval_prob = params["base_rate"]
            
            # Apply racial bias
            if races[i] in params["race_penalties"]:
                approval_prob -= params["race_penalties"][races[i]]
            
            # Apply gender bias
            if genders[i] in params["gender_penalties"]:
                approval_prob -= params["gender_penalties"][genders[i]]
            
            # Apply age bias
            if ages[i] < 25:
                approval_prob -= params["age_penalties"]["young"]
            elif ages[i] > 60:
                approval_prob -= params["age_penalties"]["old"]
            
            # Clip to valid probability range
            approval_prob = np.clip(approval_prob, 0.05, 0.95)
            
            # Generate prediction
            pred = 1 if random.random() < approval_prob else 0
            predictions.append(pred)
            
            # Generate ground truth (slightly different from prediction)
            # Simulates model being ~75% accurate
            if random.random() < 0.75:
                ground_truth = pred
            else:
                ground_truth = 1 - pred
            ground_truths.append(ground_truth)
        
        # Generate timestamps (spread over last 30 days)
        timestamps = [
            datetime.now() - timedelta(days=random.randint(0, 30))
            for _ in range(self.n_samples)
        ]
        
        # Generate optional feature data
        credit_scores = np.random.normal(690, 75, self.n_samples).astype(int)
        credit_scores = np.clip(credit_scores, 300, 850)
        
        annual_incomes = np.random.lognormal(10.85, 0.55, self.n_samples).astype(int)
        annual_incomes = np.clip(annual_incomes, 20000, 400000)
        
        # Create DataFrame
        df = pd.DataFrame({
            'prediction': predictions,
            'race': races,
            'gender': genders,
            'age': ages,
            'marital_status': marital_statuses,
            'ground_truth': ground_truths,
            'timestamp': timestamps,
            'credit_score': credit_scores,
            'annual_income': annual_incomes
        })
        
        # Calculate and print statistics
        self._print_statistics(df, bias_level)
        
        return df
    
    def _print_statistics(self, df, bias_level):
        """Print bias statistics for verification"""
        
        overall_rate = df['prediction'].mean()
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total Samples: {len(df):,}")
        print(f"   Overall Approval Rate: {overall_rate:.2%}")
        
        print(f"\n🎯 Disparate Impact Analysis:")
        
        # Race disparate impact
        print(f"\n   Race:")
        white_rate = df[df['race'] == 'White']['prediction'].mean()
        print(f"   - White:    {white_rate:.2%} (n={len(df[df['race'] == 'White']):,})")
        
        for race in ['Black', 'Hispanic', 'Asian']:
            race_rate = df[df['race'] == race]['prediction'].mean()
            race_count = len(df[df['race'] == race])
            di = race_rate / white_rate if white_rate > 0 else 0
            
            # Status indicator
            if bias_level == "none":
                status = "✅" if 0.95 <= di <= 1.05 else "⚠️"
            elif bias_level == "borderline":
                status = "✅" if 0.78 <= di <= 0.82 else "⚠️"
            elif bias_level == "moderate":
                status = "✅" if 0.68 <= di <= 0.72 else "⚠️"
            else:  # severe
                status = "✅" if 0.58 <= di <= 0.65 else "⚠️"
            
            print(f"   - {race:8} {race_rate:.2%} (n={race_count:,}) | DI: {di:.4f} {status}")
        
        # Gender disparate impact
        print(f"\n   Gender:")
        male_rate = df[df['gender'] == 'Male']['prediction'].mean()
        female_rate = df[df['gender'] == 'Female']['prediction'].mean()
        gender_di = female_rate / male_rate if male_rate > 0 else 0
        
        print(f"   - Male:   {male_rate:.2%} (n={len(df[df['gender'] == 'Male']):,})")
        print(f"   - Female: {female_rate:.2%} (n={len(df[df['gender'] == 'Female']):,})")
        print(f"   - DI (F/M): {gender_di:.4f}")
        
        # Statistical parity
        print(f"\n📐 Statistical Parity:")
        race_sp = abs(white_rate - df[df['race'] == 'Black']['prediction'].mean())
        gender_sp = abs(male_rate - female_rate)
        
        print(f"   - Race (W-B):     {race_sp:.4f} ({race_sp*100:.2f}%)")
        print(f"   - Gender (M-F):   {gender_sp:.4f} ({gender_sp*100:.2f}%)")
        
        print("="*70)
    
    def generate_and_save(self, bias_level="borderline", filename=None):
        """Generate and save predictions to CSV"""
        
        if filename is None:
            filename = f"test_predictions_{bias_level}_{self.n_samples}.csv"
        
        df = self.generate_predictions(bias_level)
        
        # Save to CSV
        output_dir = Path("test_data")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        print(f"\n✅ Saved to: {output_path}")
        print(f"   File size: {file_size_mb:.2f} MB")
        print(f"\n📤 Upload to BiasGuard:")
        print(f'   curl -X POST "http://localhost:8001/api/v1/monitor/upload_csv?model_id=YOUR_MODEL_ID" \\')
        print(f'     -F "file=@{output_path}"')
        
        return df, str(output_path)


def generate_all_test_datasets(samples_per_level=5000):
    """Generate complete test suite with all bias levels"""
    
    print("\n" + "="*70)
    print("🧪 BIASGUARD 2.0 TEST DATA GENERATOR")
    print("="*70)
    print(f"Generating {samples_per_level:,} predictions per bias level...")
    
    generator = BiasedPredictionGenerator(n_samples=samples_per_level)
    
    # Generate all bias levels
    levels = ["none", "borderline", "moderate", "severe"]
    generated_files = {}
    
    for level in levels:
        print(f"\n{'='*70}")
        print(f"LEVEL: {level.upper()}")
        print("="*70)
        
        df, filepath = generator.generate_and_save(bias_level=level)
        generated_files[level] = filepath
    
    # Generate summary
    print("\n" + "="*70)
    print("✅ ALL TEST DATASETS GENERATED")
    print("="*70)
    
    summary_table = []
    for level in levels:
        df = pd.read_csv(generated_files[level])
        white_rate = df[df['race'] == 'White']['prediction'].mean()
        black_rate = df[df['race'] == 'Black']['prediction'].mean()
        di = black_rate / white_rate if white_rate > 0 else 0
        
        summary_table.append({
            "Level": level.upper(),
            "File": generated_files[level],
            "Samples": len(df),
            "DI (B/W)": f"{di:.4f}",
            "Expected Status": {
                "none": "COMPLIANT",
                "borderline": "WARNING",
                "moderate": "NON-COMPLIANT",
                "severe": "CRITICAL"
            }[level]
        })
    
    summary_df = pd.DataFrame(summary_table)
    print("\n" + summary_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("🎯 TESTING WORKFLOW:")
    print("="*70)
    print("\n1. Register a test model:")
    print('   curl -X POST http://localhost:8001/api/v1/models/register \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"model_name": "Test Model", "model_type": "classification",')
    print('          "sensitive_attributes": ["race", "gender", "age"]}\'')
    
    print("\n2. Upload each test dataset:")
    for level in levels:
        print(f'\n   {level.upper()}:')
        print(f'   curl -X POST "http://localhost:8001/api/v1/monitor/upload_csv?model_id=MODEL_ID" \\')
        print(f'     -F "file=@{generated_files[level]}"')
    
    print("\n3. Run analysis:")
    print('   curl -X POST http://localhost:8001/api/v1/analyze \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"model_id": "MODEL_ID", "period_days": 30}\'')
    
    print("\n4. Generate report:")
    print('   curl -X POST http://localhost:8001/api/v1/reports/generate/MODEL_ID \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"report_type": "compliance", "include_recommendations": true}\'')
    
    print("\n" + "="*70)
    
    return generated_files


def generate_large_production_dataset(n_samples=50000, bias_level="borderline"):
    """
    Generate large dataset simulating real production traffic
    
    Use this to test:
    - Performance at scale
    - Drift detection over time
    - Temporal bias patterns
    """
    
    print(f"\n{'='*70}")
    print(f"Generating LARGE production dataset: {n_samples:,} samples")
    print(f"Bias Level: {bias_level}")
    print("="*70)
    
    generator = BiasedPredictionGenerator(n_samples=n_samples)
    df = generator.generate_predictions(bias_level)
    
    # Add more realistic production features
    
    # Transaction IDs
    df['transaction_id'] = [f"TXN_{uuid.uuid4().hex[:12]}" for _ in range(n_samples)]
    
    # Customer IDs (some repeat customers)
    num_customers = int(n_samples * 0.7)  # 70% unique customers, 30% repeat
    customer_pool = [f"CUST_{uuid.uuid4().hex[:8]}" for _ in range(num_customers)]
    df['customer_id'] = [random.choice(customer_pool) for _ in range(n_samples)]
    
    # Loan amounts
    df['loan_amount'] = np.random.choice(
        [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000],
        size=n_samples,
        p=[0.15, 0.20, 0.18, 0.15, 0.12, 0.10, 0.06, 0.04]
    )
    
    # Confidence scores (realistic distribution)
    confidence_scores = []
    for pred in df['prediction']:
        if pred == 1:
            # Approved - higher confidence
            conf = np.random.beta(8, 2) * 0.5 + 0.5  # Range: 0.5-1.0
        else:
            # Denied - lower confidence  
            conf = np.random.beta(2, 8) * 0.5  # Range: 0.0-0.5
        confidence_scores.append(conf)
    
    df['prediction_proba'] = confidence_scores
    
    # Sort by timestamp for realistic time series
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Save
    filename = f"production_predictions_{bias_level}_{n_samples}.csv"
    output_dir = Path("test_data")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / filename
    
    df.to_csv(output_path, index=False)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n✅ Large dataset saved: {output_path}")
    print(f"   Samples: {len(df):,}")
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df, str(output_path)


def generate_minimal_test_csv():
    """Generate minimal CSV for quick testing (100 samples)"""
    
    print("\n" + "="*70)
    print("Generating MINIMAL test CSV (100 samples)")
    print("="*70)
    
    generator = BiasedPredictionGenerator(n_samples=100)
    df = generator.generate_predictions(bias_level="moderate")
    
    # Keep only essential columns
    df = df[['prediction', 'race', 'gender', 'age', 'ground_truth']]
    
    output_path = Path("test_data") / "minimal_test.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Minimal test CSV: {output_path}")
    print(f"   Perfect for quick smoke tests!")
    
    return df, str(output_path)


def create_csv_with_known_violations():
    """
    Create CSV with KNOWN violations for unit testing
    
    This ensures specific bias patterns exist for testing detection accuracy
    """
    
    print("\n" + "="*70)
    print("Creating CSV with GUARANTEED violations")
    print("="*70)
    
    n = 2000
    
    # Create data where Black applicants have EXACTLY 0.79 approval rate
    # and White applicants have EXACTLY 1.00 approval rate
    
    white_samples = int(n * 0.6)
    black_samples = int(n * 0.2)
    hispanic_samples = int(n * 0.15)
    asian_samples = n - white_samples - black_samples - hispanic_samples
    
    data = []
    
    # White: 100% approval
    for i in range(white_samples):
        data.append({
            'prediction': 1,
            'race': 'White',
            'gender': np.random.choice(['Male', 'Female']),
            'age': np.random.randint(25, 60),
            'ground_truth': 1
        })
    
    # Black: 79% approval (EXACTLY 0.79 DI)
    black_approvals = int(black_samples * 0.79)
    for i in range(black_approvals):
        data.append({
            'prediction': 1,
            'race': 'Black',
            'gender': np.random.choice(['Male', 'Female']),
            'age': np.random.randint(25, 60),
            'ground_truth': 1
        })
    for i in range(black_samples - black_approvals):
        data.append({
            'prediction': 0,
            'race': 'Black',
            'gender': np.random.choice(['Male', 'Female']),
            'age': np.random.randint(25, 60),
            'ground_truth': 0
        })
    
    # Hispanic: 85% approval
    hispanic_approvals = int(hispanic_samples * 0.85)
    for i in range(hispanic_approvals):
        data.append({
            'prediction': 1,
            'race': 'Hispanic',
            'gender': np.random.choice(['Male', 'Female']),
            'age': np.random.randint(25, 60),
            'ground_truth': 1
        })
    for i in range(hispanic_samples - hispanic_approvals):
        data.append({
            'prediction': 0,
            'race': 'Hispanic',
            'gender': np.random.choice(['Male', 'Female']),
            'age': np.random.randint(25, 60),
            'ground_truth': 0
        })
    
    # Asian: 105% of white rate
    for i in range(asian_samples):
        data.append({
            'prediction': 1,
            'race': 'Asian',
            'gender': np.random.choice(['Male', 'Female']),
            'age': np.random.randint(25, 60),
            'ground_truth': 1
        })
    
    df = pd.DataFrame(data)
    
    # Verify exact DI
    white_rate = df[df['race'] == 'White']['prediction'].mean()
    black_rate = df[df['race'] == 'Black']['prediction'].mean()
    actual_di = black_rate / white_rate
    
    print(f"\n✅ Guaranteed Violation Dataset:")
    print(f"   White approval: {white_rate:.2%}")
    print(f"   Black approval: {black_rate:.2%}")
    print(f"   Disparate Impact: {actual_di:.4f}")
    print(f"   Expected: 0.7900 | Actual: {actual_di:.4f} ✅")
    
    # Save
    output_path = Path("test_data") / "guaranteed_violation.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n💾 Saved: {output_path}")
    print(f"   This CSV will ALWAYS trigger a violation in BiasGuard!")
    
    return df, str(output_path)


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    import sys
    
    print("\n🔬 BiasGuard 2.0 - Prediction Test Data Generator")
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "all":
            # Generate all bias levels
            generate_all_test_datasets(samples_per_level=5000)
            
        elif command == "large":
            # Generate large production dataset
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
            level = sys.argv[3] if len(sys.argv) > 3 else "borderline"
            generate_large_production_dataset(n_samples=n, bias_level=level)
            
        elif command == "minimal":
            # Generate minimal test CSV
            generate_minimal_test_csv()
            
        elif command == "guaranteed":
            # Generate guaranteed violation
            create_csv_with_known_violations()
            
        elif command in ["none", "borderline", "moderate", "severe"]:
            # Generate specific bias level
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
            generator = BiasedPredictionGenerator(n_samples=n)
            generator.generate_and_save(bias_level=command)
        
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  all          - Generate all bias levels (5K each)")
            print("  large [n]    - Generate large dataset (default: 50K)")
            print("  minimal      - Generate minimal test (100 samples)")
            print("  guaranteed   - Generate guaranteed violation")
            print("  none [n]     - Generate no bias dataset")
            print("  borderline [n] - Generate borderline bias")
            print("  moderate [n] - Generate moderate bias")
            print("  severe [n]   - Generate severe bias")
    
    else:
        # Default: Generate all test datasets
        print("\nNo arguments provided. Generating all test datasets...\n")
        generate_all_test_datasets(samples_per_level=5000)