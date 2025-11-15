"""
BiasGuard Bias Injection Tester
Artificially inject bias into fair model predictions to test detection

This allows you to test BiasGuard's detection capabilities without
retraining models with sensitive attributes as features.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import uuid

def inject_borderline_bias(y_pred, sensitive_data, injection_level='borderline'):
    """
    Inject bias into predictions to test BiasGuard detection
    
    Args:
        y_pred: Original fair predictions (numpy array)
        sensitive_data: Dict of sensitive attributes
        injection_level: 'borderline' (DI~0.80), 'moderate' (DI~0.70), 'severe' (DI~0.60)
    
    Returns:
        biased_predictions: Modified predictions with injected bias
    """
    
    print(f"\n🔬 Injecting {injection_level} bias into predictions...")
    print("="*70)
    
    biased_pred = y_pred.copy()
    
    # Define flip rates based on injection level
    flip_rates = {
        'borderline': {
            'Black': 0.12,      # Flip 12% of Black approvals to denials
            'Hispanic': 0.08,
            'Female': 0.06,
            'Young': 0.05       # Age < 25
        },
        'moderate': {
            'Black': 0.20,      # Flip 20% of Black approvals to denials
            'Hispanic': 0.15,
            'Female': 0.12,
            'Young': 0.10
        },
        'severe': {
            'Black': 0.30,      # Flip 30% of Black approvals to denials
            'Hispanic': 0.22,
            'Female': 0.18,
            'Young': 0.15
        }
    }
    
    rates = flip_rates[injection_level]
    
    # Track changes
    changes = {
        'race': 0,
        'gender': 0,
        'age': 0
    }
    
    # Inject racial bias
    if 'race' in sensitive_data:
        race = sensitive_data['race']
        for idx in range(len(biased_pred)):
            if biased_pred[idx] == 1:  # Only flip approvals to denials
                if race[idx] == 'Black' and np.random.random() < rates['Black']:
                    biased_pred[idx] = 0
                    changes['race'] += 1
                elif race[idx] == 'Hispanic' and np.random.random() < rates['Hispanic']:
                    biased_pred[idx] = 0
                    changes['race'] += 1
    
    # Inject gender bias
    if 'gender' in sensitive_data:
        gender = sensitive_data['gender']
        for idx in range(len(biased_pred)):
            if biased_pred[idx] == 1:
                if gender[idx] == 'Female' and np.random.random() < rates['Female']:
                    biased_pred[idx] = 0
                    changes['gender'] += 1
    
    # Inject age bias
    if 'age' in sensitive_data:
        age = sensitive_data['age']
        for idx in range(len(biased_pred)):
            if biased_pred[idx] == 1:
                if age[idx] < 25 and np.random.random() < rates['Young']:
                    biased_pred[idx] = 0
                    changes['age'] += 1
    
    total_flipped = sum(changes.values())
    original_approvals = np.sum(y_pred == 1)
    
    print(f"Original approval rate: {np.mean(y_pred):.2%}")
    print(f"New approval rate: {np.mean(biased_pred):.2%}")
    print(f"\nPredictions flipped: {total_flipped:,} / {len(y_pred):,} ({total_flipped/len(y_pred):.2%})")
    print(f"  - Race bias: {changes['race']:,} flips")
    print(f"  - Gender bias: {changes['gender']:,} flips")
    print(f"  - Age bias: {changes['age']:,} flips")
    
    # Calculate expected disparate impact
    if 'race' in sensitive_data:
        race = sensitive_data['race']
        white_rate = np.mean(biased_pred[race == 'White'])
        black_rate = np.mean(biased_pred[race == 'Black'])
        expected_di = black_rate / white_rate if white_rate > 0 else 0
        print(f"\n📊 Expected Disparate Impact (Black/White): {expected_di:.4f}")
        
        if injection_level == 'borderline':
            if 0.78 <= expected_di <= 0.82:
                print("   ✅ BORDERLINE bias achieved (0.78-0.82)")
            else:
                print(f"   ⚠️  Outside target range, got {expected_di:.4f}")
    
    print("="*70)
    
    return biased_pred


def create_biased_model_for_testing(original_model_id, injection_level='borderline'):
    """
    Load an existing fair model and create a biased version for testing
    
    This simulates what would happen if the model had discriminatory bias
    without actually retraining with sensitive attributes
    """
    
    print(f"\n🎯 Creating biased test model from {original_model_id}")
    print(f"Injection level: {injection_level}")
    print("="*70)
    
    # Import here to avoid circular imports
    from api.routes.training import get_model_data, store_model_data
    from api.models.database import get_db
    
    db = next(get_db())
    
    # Get original model data
    model_data = get_model_data(original_model_id, db)
    
    if not model_data:
        print(f"❌ Model {original_model_id} not found!")
        return None
    
    # Get predictions and sensitive data
    y_pred = model_data['y_pred']
    y_test = model_data['y_test']
    
    # Get sensitive data
    if 'sensitive_data' in model_data:
        sensitive_data = model_data['sensitive_data']
    else:
        s_test = model_data['s_test']
        sensitive_columns = model_data['sensitive_columns']
        
        sensitive_data = {}
        if isinstance(sensitive_columns, list) and len(sensitive_columns) > 1:
            for i, col in enumerate(sensitive_columns):
                sensitive_data[col] = s_test[:, i]
        else:
            col_name = sensitive_columns[0] if isinstance(sensitive_columns, list) else sensitive_columns
            sensitive_data = {col_name: s_test}
    
    # Inject bias into predictions
    biased_pred = inject_borderline_bias(y_pred, sensitive_data, injection_level)
    
    # Create new model ID
    new_model_id = f"model_biased_{injection_level}_{uuid.uuid4().hex[:8]}"
    
    # Store biased model data
    biased_model_data = model_data.copy()
    biased_model_data['y_pred'] = biased_pred
    biased_model_data['bias_injected'] = True
    biased_model_data['injection_level'] = injection_level
    biased_model_data['original_model_id'] = original_model_id
    
    store_model_data(new_model_id, biased_model_data)
    
    # Create database entry
    from api.models import schemas, crud
    from sklearn.metrics import accuracy_score
    
    new_accuracy = accuracy_score(y_test, biased_pred)
    
    biased_model_db = schemas.ModelCreate(
        model_id=new_model_id,
        model_type=f"{model_data['model_type']} (Biased Test)",
        task_type=model_data['task_type'],
        dataset_name=model_data.get('file_id', 'test_dataset'),
        target_column=model_data['target_column'],
        sensitive_columns=model_data['sensitive_columns'],
        feature_count=model_data['X_test'].shape[1],
        training_samples=len(model_data.get('X_train', [])) if model_data.get('X_train') is not None else 0,
        test_samples=len(model_data['X_test']),
        accuracy=float(new_accuracy),
        mlflow_run_id=model_data.get('mlflow_run_id', 'test_run')
    )
    
    crud.create_model(db, biased_model_db)
    
    print(f"\n✅ Biased test model created: {new_model_id}")
    print(f"   Original accuracy: {model_data.get('accuracy', 0):.2%}")
    print(f"   New accuracy: {new_accuracy:.2%}")
    print(f"   Accuracy drop: {(model_data.get('accuracy', 0) - new_accuracy):.2%}")
    print("\n🎯 Now run bias detection on this model to test BiasGuard!")
    print(f"   POST /api/v1/bias")
    print(f"   {{'model_id': '{new_model_id}'}}")
    print("="*70)
    
    return new_model_id


def test_biasguard_detection():
    """
    Full workflow to test BiasGuard's detection capabilities
    """
    
    print("\n" + "="*70)
    print("🧪 BIASGUARD DETECTION TEST SUITE")
    print("="*70)
    
    from api.models.database import get_db, Model
    from sqlalchemy.orm import Session
    
    db = next(get_db())
    
    # Get most recent fair models directly from database
    models = db.query(Model).order_by(Model.created_at.desc()).limit(5).all()
    
    if not models:
        print("❌ No models found! Train a model first.")
        return
    
    print(f"\n📋 Found {len(models)} recent models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model.model_id} - {model.model_type} (Accuracy: {model.accuracy:.2%})")
    
    # Use most recent model
    original_model = models[0]
    print(f"\n🎯 Using: {original_model.model_id}")
    
    # Create test cases with different bias levels
    test_cases = [
        ('borderline', 'DI ~ 0.79-0.81'),
        ('moderate', 'DI ~ 0.68-0.72'),
        ('severe', 'DI ~ 0.58-0.62')
    ]
    
    created_models = []
    
    for level, description in test_cases:
        print(f"\n{'='*70}")
        print(f"Creating {level.upper()} bias test case ({description})")
        print("="*70)
        
        new_model_id = create_biased_model_for_testing(
            original_model.model_id,
            injection_level=level
        )
        
        if new_model_id:
            created_models.append((new_model_id, level))
    
    # Summary
    print("\n" + "="*70)
    print("✅ TEST MODELS CREATED")
    print("="*70)
    print("\nRun bias detection on these models:")
    for model_id, level in created_models:
        print(f"\n  {level.upper()}:")
        print(f"  curl -X POST http://localhost:8001/api/v1/bias \\")
        print(f"    -H 'Content-Type: application/json' \\")
        print(f"    -d '{{'model_id': '{model_id}'}}'")
    
    print("\n" + "="*70)
    print("🎯 EXPECTED RESULTS:")
    print("="*70)
    print("  BORDERLINE: Should show 'WARNING' status, DI ~ 0.80")
    print("  MODERATE:   Should show 'NON-COMPLIANT', DI ~ 0.70")
    print("  SEVERE:     Should show 'CRITICAL', DI ~ 0.60")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            # Run full test suite
            test_biasguard_detection()
        else:
            # Create biased version of specific model
            model_id = sys.argv[1]
            level = sys.argv[2] if len(sys.argv) > 2 else 'borderline'
            create_biased_model_for_testing(model_id, level)
    else:
        print("Usage:")
        print("  python bias_injection_tester.py test                    # Run full test suite")
        print("  python bias_injection_tester.py MODEL_ID [level]        # Create biased version")
        print("\nLevels: borderline, moderate, severe")