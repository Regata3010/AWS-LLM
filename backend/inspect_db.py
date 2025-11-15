import sqlite3
import json
from datetime import datetime

def inspect_database():
    """Quick database inspection script"""
    
    conn = sqlite3.connect('/Users/AWS-LLM/backend/biasguard.db')
    conn.row_factory = sqlite3.Row 
    cursor = conn.cursor()
    
   
    print("BIASGUARD DATABASE INSPECTION")
    
    
    # Models
    print("\nMODELS:")
    cursor.execute("SELECT * FROM models ORDER BY created_at DESC")
    models = cursor.fetchall()
    
    for model in models:
        print(f"\n  Model ID: {model['model_id']}")
        print(f"  Type: {model['model_type']} ({model['task_type']})")
        print(f"  Accuracy: {model['accuracy']:.4f}")
        print(f"  Bias Status: {model['bias_status']}")
        print(f"  Created: {model['created_at']}")
        print(f"  Sensitive Cols: {model['sensitive_columns']}")
    
    # Summary
    print("\n")
    print("SUMMARY STATISTICS:")
    
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN bias_status = 'compliant' THEN 1 ELSE 0 END) as compliant,
            SUM(CASE WHEN bias_status = 'warning' THEN 1 ELSE 0 END) as warning,
            SUM(CASE WHEN bias_status = 'critical' THEN 1 ELSE 0 END) as critical,
            SUM(CASE WHEN bias_status = 'unknown' THEN 1 ELSE 0 END) as unknown,
            AVG(accuracy) as avg_accuracy
        FROM models
    """)
    
    stats = cursor.fetchone()
    print(f"  Total Models: {stats['total']}")
    print(f"  Compliant: {stats['compliant']}")
    print(f"  Warning: {stats['warning']}")
    print(f"  Critical: {stats['critical']}")
    print(f"  Unknown: {stats['unknown']}")
    print(f"  Avg Accuracy: {stats['avg_accuracy'] if stats['avg_accuracy'] else 0}")
    
    # Bias Analyses
    print("\nBIAS ANALYSES:")
    cursor.execute("SELECT COUNT(*) as total FROM bias_analyses")
    print(f"  Total Analyses: {cursor.fetchone()['total']}")
    
    cursor.execute("""
        SELECT model_id, compliance_status, analyzed_at 
        FROM bias_analyses 
        ORDER BY analyzed_at DESC 
        LIMIT 5
    """)
    
    for analysis in cursor.fetchall():
        print(f"  - {analysis['model_id']}: {analysis['compliance_status']} ({analysis['analyzed_at']})")
    
    # Mitigations
    print("\nMITIGATIONS:")
    cursor.execute("SELECT COUNT(*) as total FROM mitigations")
    print(f"  Total Mitigations: {cursor.fetchone()['total']}")
    
    cursor.execute("""
        SELECT 
            original_model_id, 
            strategy, 
            original_accuracy,
            new_accuracy,
            accuracy_impact 
        FROM mitigations 
        ORDER BY created_at DESC
    """)
    
    for mitigation in cursor.fetchall():
        print(f"  - {mitigation['original_model_id']}: {mitigation['strategy']}")
        print(f"    Accuracy: {mitigation['original_accuracy']:.4f} → {mitigation['new_accuracy']:.4f} (Δ {mitigation['accuracy_impact']:.4f})")
    
    conn.close()
    print("\n")

if __name__ == "__main__":
    inspect_database()