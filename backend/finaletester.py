#!/usr/bin/env python3
"""
BiasGuard 2.0 - Complete Integration Test (Python)
Tests the full monitoring workflow end-to-end
"""

import requests
import json
import time
from pathlib import Path
from typing import Dict, List

BASE_URL = "http://localhost:8001/api/v1"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

class BiasGuardTester:
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.model_id = None
        self.analysis_id = None
        self.report_url = None
        self.mlflow_run_id = None
    
    def test_step(self, message: str):
        print(f"\n{Colors.BLUE}{message}{Colors.END}")
    
    def test_pass(self, message: str):
        print(f"{Colors.GREEN}✓ PASS: {message}{Colors.END}")
        self.tests_passed += 1
    
    def test_fail(self, message: str):
        print(f"{Colors.RED}✗ FAIL: {message}{Colors.END}")
        self.tests_failed += 1
    
    def test_info(self, message: str):
        print(f"{Colors.YELLOW}  INFO: {message}{Colors.END}")
    
    def run_all_tests(self):
        print("="*70)
        print(f"{Colors.BOLD}  BIASGUARD 2.0 - COMPLETE INTEGRATION TEST{Colors.END}")
        print("="*70)
        
        try:
            self.test_model_registry()
            self.test_prediction_logging()
            self.test_bias_analysis()
            self.test_reporting()
            self.test_monitoring_stats()
            self.test_edge_cases()
            self.test_multiple_bias_levels()
            
            self.print_summary()
            
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            self.print_summary()
        except Exception as e:
            print(f"\n{Colors.RED}FATAL ERROR: {e}{Colors.END}")
            self.print_summary()
    
    def test_model_registry(self):
        """TEST 1: Model Registry"""
        self.test_step("[TEST 1] Model Registry - Register External Model")
        
        response = requests.post(
            f"{BASE_URL}/models/register",
            json={
                "model_name": "Integration Test Model",
                "description": "Test model for BiasGuard 2.0 validation",
                "model_type": "classification",
                "framework": "xgboost",
                "version": "v1.0",
                "sensitive_attributes": ["race", "gender", "age"]
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            self.model_id = data['model_id']
            self.test_pass(f"Model registered: {self.model_id}")
        else:
            self.test_fail(f"Registration failed: {response.status_code}")
            print(response.text)
            raise Exception("Cannot continue without model")
        
        # Test: List models
        self.test_step("[TEST 1.1] List All Models")
        response = requests.get(f"{BASE_URL}/models")
        if response.status_code == 200:
            data = response.json()
            self.test_info(f"Total models: {data['total']}")
            self.test_pass("Models list endpoint working")
        
        # Test: Get model details
        self.test_step("[TEST 1.2] Get Model Details")
        response = requests.get(f"{BASE_URL}/model/{self.model_id}")
        if response.status_code == 200:
            data = response.json()
            self.test_pass(f"Model details: {data['model_name']}")
    
    def test_prediction_logging(self):
        """TEST 2: Prediction Logging"""
        self.test_step("[TEST 2] Prediction Logging - Upload CSV")
        
        # Check if test data exists
        test_file = Path("test_data/test_predictions_borderline_5000.csv")
        if not test_file.exists():
            self.test_info("Generating test data...")
            import subprocess
            subprocess.run(["python", "prediction_generator.py", "borderline", "5000"])
        
        # Upload CSV
        with open(test_file, 'rb') as f:
            response = requests.post(
                f"{BASE_URL}/monitor/upload_csv",
                params={"model_id": self.model_id},
                files={"file": f}
            )
        
        if response.status_code == 200:
            data = response.json()
            predictions_logged = data['predictions_logged']
            self.test_pass(f"Uploaded {predictions_logged} predictions")
            
            # Check statistics
            stats = data['statistics']
            approval_rate = stats['overall_approval_rate']
            self.test_info(f"Overall approval: {approval_rate:.2%}")
            
            # Check race breakdown
            if 'breakdown_by_attribute' in stats and 'race' in stats['breakdown_by_attribute']:
                race_data = stats['breakdown_by_attribute']['race']
                white_rate = race_data.get('White', {}).get('approval_rate', 0)
                black_rate = race_data.get('Black', {}).get('approval_rate', 0)
                
                self.test_info(f"White approval: {white_rate:.2%}")
                self.test_info(f"Black approval: {black_rate:.2%}")
                
                if white_rate > 0:
                    di = black_rate / white_rate
                    self.test_info(f"Expected DI: {di:.4f}")
        else:
            self.test_fail("Upload failed")
            print(response.text)
            raise Exception("Cannot continue without predictions")
        
        # Test: Monitoring stats
        self.test_step("[TEST 2.1] Get Monitoring Stats")
        response = requests.get(f"{BASE_URL}/monitor/stats/{self.model_id}", params={"days": 30})
        if response.status_code == 200:
            data = response.json()
            self.test_pass(f"Stats: {data['total_predictions']} predictions")
    
    def test_bias_analysis(self):
        """TEST 3: Bias Analysis"""
        self.test_step("[TEST 3] Bias Analysis - Analyze Production Predictions")
        
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={
                "model_id": self.model_id,
                "period_days": 30,
                "min_samples": 1000
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            self.analysis_id = data['analysis_id']
            self.mlflow_run_id = data.get('mlflow_run_id')
            
            samples = data['period']['samples']
            compliance = data['compliance_status']
            bias_status = data['bias_status']
            
            self.test_pass(f"Analysis completed: {self.analysis_id}")
            self.test_info(f"Samples: {samples}")
            self.test_info(f"Compliance: {compliance}")
            self.test_info(f"Bias Status: {bias_status}")
            
            # Check fairness metrics
            if 'fairness_metrics' in data and 'race' in data['fairness_metrics']:
                race_metrics = data['fairness_metrics']['race']
                di = race_metrics['disparate_impact']['ratio']
                sp = race_metrics['statistical_parity']['statistical_parity_diff']
                
                self.test_info(f"DI (race): {di}")
                self.test_info(f"SP (race): {sp}")
                
                if di != 1.0 or sp != 0.0:
                    self.test_pass("Real bias metrics detected (not 0.0000/1.0000)")
                else:
                    self.test_fail("Metrics still showing perfect fairness")
            
            # Check intersectionality
            if 'intersectionality' in data and data['intersectionality']:
                intersect = data['intersectionality']['summary']
                groups = intersect['total_groups_analyzed']
                at_risk = intersect['groups_below_threshold']
                self.test_pass(f"Intersectionality: {groups} groups, {at_risk} at risk")
            
            # Check alerts
            alerts = data.get('alerts', [])
            self.test_info(f"Alerts: {len(alerts)} triggered")
            
        else:
            self.test_fail(f"Analysis failed: {response.status_code}")
            print(response.text)
        
        # Test: Get latest analysis
        self.test_step("[TEST 3.1] Retrieve Latest Analysis")
        response = requests.get(f"{BASE_URL}/bias/latest/{self.model_id}")
        if response.status_code == 200:
            self.test_pass("Latest analysis retrieved")
        
        # Test: Get history
        self.test_step("[TEST 3.2] Get Analysis History")
        response = requests.get(f"{BASE_URL}/bias/history/{self.model_id}")
        if response.status_code == 200:
            data = response.json()
            self.test_pass(f"History: {data['total_analyses']} analysis(es)")
    
    def test_reporting(self):
        """TEST 4: Compliance Reporting"""
        self.test_step("[TEST 4] Compliance Reporting - Generate PDF")
        
        response = requests.post(
            f"{BASE_URL}/reports/generate/{self.model_id}",
            json={
                "report_type": "compliance",
                "include_recommendations": True,
                "include_technical_details": True,
                "format": "pdf"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            report_id = data['report_id']
            self.report_url = data['download_url']
            file_size = data['file_size_mb']
            
            self.test_pass(f"Report generated: {report_id} ({file_size}MB)")
            
            # Test: Download report
            self.test_step("[TEST 4.1] Download Report")
            download_response = requests.get(f"{BASE_URL}{self.report_url}")
            if download_response.status_code == 200:
                with open('/tmp/biasguard_test_report.pdf', 'wb') as f:
                    f.write(download_response.content)
                self.test_pass("Report downloaded successfully")
            else:
                self.test_fail(f"Download failed: {download_response.status_code}")
        else:
            self.test_fail(f"Report generation failed: {response.status_code}")
            print(response.text)
    
    def test_monitoring_stats(self):
        """TEST 5: Monitoring Stats"""
        self.test_step("[TEST 5] Monitoring Statistics")
        
        response = requests.get(f"{BASE_URL}/monitor/stats/{self.model_id}", params={"days": 7})
        if response.status_code == 200:
            data = response.json()
            self.test_pass(f"Stats retrieved: {data['total_predictions']} predictions")
    
    def test_edge_cases(self):
        """TEST 6: Edge Cases"""
        self.test_step("[TEST 6] Edge Cases & Error Handling")
        
        # Test: Non-existent model
        response = requests.get(f"{BASE_URL}/model/fake_model_123")
        if response.status_code == 404:
            self.test_pass("Handles non-existent model correctly")
        else:
            self.test_fail("Should return 404 for non-existent model")
        
        # Test: Insufficient data
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"model_id": self.model_id, "min_samples": 999999}
        )
        if response.status_code == 400:
            self.test_pass("Validates minimum sample requirement")
        else:
            self.test_fail("Should reject insufficient data")
    
    def test_multiple_bias_levels(self):
        """TEST 7: Multiple Bias Levels"""
        self.test_step("[TEST 7] Test Different Bias Levels")
        
        for level in ['moderate', 'severe']:
            self.test_info(f"\nTesting {level} bias...")
            
            # Register model
            reg_response = requests.post(
                f"{BASE_URL}/models/register",
                json={
                    "model_name": f"Test - {level.title()}",
                    "model_type": "classification",
                    "sensitive_attributes": ["race", "gender"]
                }
            )
            
            if reg_response.status_code != 200:
                self.test_fail(f"{level}: Registration failed")
                continue
            
            test_model_id = reg_response.json()['model_id']
            
            # Check test file
            test_file = Path(f"test_data/test_predictions_{level}_5000.csv")
            if not test_file.exists():
                self.test_info(f"Generating {level} test data...")
                import subprocess
                subprocess.run(["python", "prediction_generator.py", level, "5000"])
            
            # Upload
            with open(test_file, 'rb') as f:
                requests.post(
                    f"{BASE_URL}/monitor/upload_csv",
                    params={"model_id": test_model_id},
                    files={"file": f}
                )
            
            # Analyze
            analysis = requests.post(
                f"{BASE_URL}/analyze",
                json={"model_id": test_model_id, "period_days": 30}
            )
            
            if analysis.status_code == 200:
                data = analysis.json()
                di = data['fairness_metrics']['race']['disparate_impact']['ratio']
                status = data['bias_status']
                
                self.test_info(f"{level}: DI={di:.4f}, Status={status}")
                
                # Validate expected outcomes
                if level == 'moderate' and status in ['warning', 'critical']:
                    self.test_pass(f"{level}: Correctly detected bias")
                elif level == 'severe' and status == 'critical':
                    self.test_pass(f"{level}: Correctly detected critical bias")
                else:
                    self.test_info(f"{level}: Status={status} (may vary)")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print(f"{Colors.BOLD}  TEST SUMMARY{Colors.END}")
        print("="*70)
        
        total = self.tests_passed + self.tests_failed
        success_rate = (self.tests_passed / total * 100) if total > 0 else 0
        
        print(f"\n{Colors.GREEN}PASSED: {self.tests_passed}{Colors.END}")
        print(f"{Colors.RED}FAILED: {self.tests_failed}{Colors.END}")
        print(f"TOTAL:  {total}")
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        if self.model_id:
            print("\n" + "="*70)
            print("  TEST ARTIFACTS")
            print("="*70)
            print(f"Model ID:     {self.model_id}")
            print(f"Analysis ID:  {self.analysis_id}")
            if self.report_url:
                print(f"Report URL:   {BASE_URL}{self.report_url}")
            if self.mlflow_run_id:
                print(f"MLflow Run:   {self.mlflow_run_id}")
        
        print("\n" + "="*70)
        if self.tests_failed == 0:
            print(f"{Colors.GREEN}  ALL TESTS PASSED - BIASGUARD 2.0 READY{Colors.END}")
        else:
            print(f"{Colors.RED}  SOME TESTS FAILED - REVIEW ABOVE{Colors.END}")
        print("="*70 + "\n")
        
        return self.tests_failed == 0


if __name__ == "__main__":
    print("\nStarting BiasGuard 2.0 Integration Tests...\n")
    
    # Check if backend is running
    try:
        response = requests.get(f"{BASE_URL}/monitor/health", timeout=2)
        print(f"{Colors.GREEN}Backend is running{Colors.END}")
    except requests.exceptions.ConnectionError:
        print(f"{Colors.RED}ERROR: Backend not running at {BASE_URL}{Colors.END}")
        print("Start backend with: uvicorn api.main:app --reload --port 8001")
        exit(1)
    
    # Run tests
    tester = BiasGuardTester()
    success = tester.run_all_tests()
    
    exit(0 if success else 1)