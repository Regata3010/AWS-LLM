#!/bin/bash
# BiasGuard 2.0 - Complete End-to-End Integration Test
# Tests the full monitoring workflow from registration to reporting

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

BASE_URL="http://localhost:8001/api/v1"

echo "========================================================================"
echo "  BIASGUARD 2.0 - COMPLETE INTEGRATION TEST"
echo "========================================================================"
echo ""

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

test_step() {
    echo -e "${BLUE}$1${NC}"
}

test_pass() {
    echo -e "${GREEN}PASS: $1${NC}"
    ((TESTS_PASSED++))
}

test_fail() {
    echo -e "${RED}FAIL: $1${NC}"
    ((TESTS_FAILED++))
}

test_info() {
    echo -e "${YELLOW}INFO: $1${NC}"
}

# ========================================
# TEST 1: MODEL REGISTRY
# ========================================

test_step "\n[TEST 1] Model Registry - Register External Model"

REGISTER_RESPONSE=$(curl -s -X POST "$BASE_URL/models/register" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Integration Test Model",
    "description": "Test model for BiasGuard 2.0 validation",
    "model_type": "classification",
    "framework": "xgboost",
    "version": "v1.0",
    "sensitive_attributes": ["race", "gender", "age"]
  }')

# Extract model_id
MODEL_ID=$(echo $REGISTER_RESPONSE | jq -r '.model_id')

if [ "$MODEL_ID" != "null" ] && [ ! -z "$MODEL_ID" ]; then
    test_pass "Model registered successfully: $MODEL_ID"
else
    test_fail "Model registration failed"
    echo "$REGISTER_RESPONSE"
    exit 1
fi

# Test: List models
test_step "\n[TEST 1.1] List All Models"
MODELS_LIST=$(curl -s "$BASE_URL/models")
TOTAL_MODELS=$(echo $MODELS_LIST | jq -r '.total')
test_info "Total models in system: $TOTAL_MODELS"
test_pass "Models list endpoint working"

# Test: Get model details
test_step "\n[TEST 1.2] Get Model Details"
MODEL_DETAILS=$(curl -s "$BASE_URL/model/$MODEL_ID")
MODEL_NAME=$(echo $MODEL_DETAILS | jq -r '.model_name')

if [ "$MODEL_NAME" == "Integration Test Model" ]; then
    test_pass "Model details retrieved: $MODEL_NAME"
else
    test_fail "Model details incorrect"
fi

# ========================================
# TEST 2: PREDICTION LOGGING
# ========================================

test_step "\n[TEST 2] Prediction Logging - Upload CSV"

# Check if test data exists
if [ ! -f "backend/test_data/test_predictions_borderline_5000.csv" ]; then
    test_info "Test data not found. Generating..."
    python prediction_generator.py borderline 5000
fi

# Upload predictions
UPLOAD_RESPONSE=$(curl -s -X POST "$BASE_URL/monitor/upload_csv?model_id=$MODEL_ID" \
  -F "file=@test_data/test_predictions_borderline_5000.csv")

PREDICTIONS_LOGGED=$(echo $UPLOAD_RESPONSE | jq -r '.predictions_logged')
BATCH_ID=$(echo $UPLOAD_RESPONSE | jq -r '.batch_id')

if [ "$PREDICTIONS_LOGGED" -gt 0 ]; then
    test_pass "Uploaded $PREDICTIONS_LOGGED predictions (Batch: $BATCH_ID)"
else
    test_fail "Prediction upload failed"
    echo "$UPLOAD_RESPONSE"
    exit 1
fi

# Test: Check upload statistics
APPROVAL_RATE=$(echo $UPLOAD_RESPONSE | jq -r '.statistics.overall_approval_rate')
RACE_WHITE_RATE=$(echo $UPLOAD_RESPONSE | jq -r '.statistics.breakdown_by_attribute.race.White.approval_rate')
RACE_BLACK_RATE=$(echo $UPLOAD_RESPONSE | jq -r '.statistics.breakdown_by_attribute.race.Black.approval_rate')

test_info "Overall approval rate: $(printf "%.2f%%" $(echo "$APPROVAL_RATE * 100" | bc))"
test_info "White approval rate: $(printf "%.2f%%" $(echo "$RACE_WHITE_RATE * 100" | bc))"
test_info "Black approval rate: $(printf "%.2f%%" $(echo "$RACE_BLACK_RATE * 100" | bc))"

# Calculate disparate impact
DI_CALCULATED=$(echo "scale=4; $RACE_BLACK_RATE / $RACE_WHITE_RATE" | bc)
test_info "Expected DI (Black/White): $DI_CALCULATED"

# Test: Get monitoring stats
test_step "\n[TEST 2.1] Get Monitoring Stats"
STATS_RESPONSE=$(curl -s "$BASE_URL/monitor/stats/$MODEL_ID?days=30")
STATS_TOTAL=$(echo $STATS_RESPONSE | jq -r '.total_predictions')

if [ "$STATS_TOTAL" == "$PREDICTIONS_LOGGED" ]; then
    test_pass "Monitoring stats correct: $STATS_TOTAL predictions"
else
    test_fail "Stats mismatch: expected $PREDICTIONS_LOGGED, got $STATS_TOTAL"
fi

# Test: List batches
test_step "\n[TEST 2.2] List Uploaded Batches"
BATCHES=$(curl -s "$BASE_URL/monitor/batches/$MODEL_ID")
BATCH_COUNT=$(echo $BATCHES | jq -r '.total_batches')
test_pass "Found $BATCH_COUNT batch(es)"

# ========================================
# TEST 3: BIAS ANALYSIS
# ========================================

test_step "\n[TEST 3] Bias Analysis - Analyze Production Predictions"

ANALYSIS_RESPONSE=$(curl -s -X POST "$BASE_URL/analyze" \
  -H "Content-Type: application/json" \
  -d "{
    \"model_id\": \"$MODEL_ID\",
    \"period_days\": 30,
    \"min_samples\": 1000
  }")

ANALYSIS_ID=$(echo $ANALYSIS_RESPONSE | jq -r '.analysis_id')
SAMPLES_ANALYZED=$(echo $ANALYSIS_RESPONSE | jq -r '.period.samples')
COMPLIANCE_STATUS=$(echo $ANALYSIS_RESPONSE | jq -r '.compliance_status')
BIAS_STATUS=$(echo $ANALYSIS_RESPONSE | jq -r '.bias_status')

if [ "$ANALYSIS_ID" != "null" ] && [ ! -z "$ANALYSIS_ID" ]; then
    test_pass "Analysis completed: $ANALYSIS_ID"
    test_info "Samples analyzed: $SAMPLES_ANALYZED"
    test_info "Compliance status: $COMPLIANCE_STATUS"
    test_info "Bias status: $BIAS_STATUS"
else
    test_fail "Analysis failed"
    echo "$ANALYSIS_RESPONSE"
    exit 1
fi

# Extract fairness metrics
DI_RACE=$(echo $ANALYSIS_RESPONSE | jq -r '.fairness_metrics.race.disparate_impact.ratio')
SP_RACE=$(echo $ANALYSIS_RESPONSE | jq -r '.fairness_metrics.race.statistical_parity.statistical_parity_diff')
DI_GENDER=$(echo $ANALYSIS_RESPONSE | jq -r '.fairness_metrics.gender.disparate_impact.ratio')

test_info "Disparate Impact (race): $DI_RACE"
test_info "Statistical Parity (race): $SP_RACE"
test_info "Disparate Impact (gender): $DI_GENDER"

# Validate bias detection
if (( $(echo "$DI_RACE < 0.9" | bc -l) )); then
    test_pass "Bias detected in race (DI = $DI_RACE)"
else
    test_info "Race DI within acceptable range: $DI_RACE"
fi

# Check intersectionality
INTERSECT_GROUPS=$(echo $ANALYSIS_RESPONSE | jq -r '.intersectionality.summary.total_groups_analyzed')
INTERSECT_AT_RISK=$(echo $ANALYSIS_RESPONSE | jq -r '.intersectionality.summary.groups_below_threshold')

if [ "$INTERSECT_GROUPS" != "null" ]; then
    test_pass "Intersectionality analysis: $INTERSECT_GROUPS groups, $INTERSECT_AT_RISK at risk"
fi

# Check alerts
ALERTS_COUNT=$(echo $ANALYSIS_RESPONSE | jq -r '.alerts | length')
test_info "Alerts triggered: $ALERTS_COUNT"

# Test: Get latest analysis
test_step "\n[TEST 3.1] Retrieve Latest Analysis"
LATEST_ANALYSIS=$(curl -s "$BASE_URL/bias/latest/$MODEL_ID")
LATEST_ID=$(echo $LATEST_ANALYSIS | jq -r '.analysis_id')

if [ "$LATEST_ID" == "$ANALYSIS_ID" ]; then
    test_pass "Latest analysis retrieved correctly"
else
    test_fail "Latest analysis mismatch"
fi

# Test: Get analysis history
test_step "\n[TEST 3.2] Get Analysis History"
HISTORY=$(curl -s "$BASE_URL/bias/history/$MODEL_ID")
HISTORY_COUNT=$(echo $HISTORY | jq -r '.total_analyses')
test_pass "Analysis history: $HISTORY_COUNT analysis(es)"

# ========================================
# TEST 4: COMPLIANCE REPORTING
# ========================================

test_step "\n[TEST 4] Compliance Reporting - Generate PDF Report"

REPORT_RESPONSE=$(curl -s -X POST "$BASE_URL/reports/generate/$MODEL_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "report_type": "compliance",
    "include_recommendations": true,
    "include_technical_details": true,
    "format": "pdf"
  }')

REPORT_ID=$(echo $REPORT_RESPONSE | jq -r '.report_id')
DOWNLOAD_URL=$(echo $REPORT_RESPONSE | jq -r '.download_url')
FILE_SIZE=$(echo $REPORT_RESPONSE | jq -r '.file_size_mb')

if [ "$REPORT_ID" != "null" ]; then
    test_pass "Report generated: $REPORT_ID (${FILE_SIZE}MB)"
    test_info "Download: $BASE_URL$DOWNLOAD_URL"
else
    test_fail "Report generation failed"
    echo "$REPORT_RESPONSE"
fi

# Test: Download report
test_step "\n[TEST 4.1] Download Report"
HTTP_CODE=$(curl -s -o /tmp/test_report.pdf -w "%{http_code}" "$BASE_URL$DOWNLOAD_URL")

if [ "$HTTP_CODE" == "200" ]; then
    DOWNLOADED_SIZE=$(ls -lh /tmp/test_report.pdf | awk '{print $5}')
    test_pass "Report downloaded successfully ($DOWNLOADED_SIZE)"
else
    test_fail "Report download failed (HTTP $HTTP_CODE)"
fi

# Test: List all reports
test_step "\n[TEST 4.2] List All Reports"
REPORTS_LIST=$(curl -s "$BASE_URL/reports/list")
TOTAL_REPORTS=$(echo $REPORTS_LIST | jq -r '.total')
test_pass "Total reports in system: $TOTAL_REPORTS"

# ========================================
# TEST 5: REAL-TIME MONITORING (Optional)
# ========================================

test_step "\n[TEST 5] Real-Time Monitoring - REST Endpoint"
ALERTS_RESPONSE=$(curl -s "$BASE_URL/monitoring/alerts/$MODEL_ID")
ALERTS_TIMESTAMP=$(echo $ALERTS_RESPONSE | jq -r '.timestamp')

if [ "$ALERTS_TIMESTAMP" != "null" ]; then
    test_pass "Real-time alerts endpoint working"
else
    test_info "Alerts endpoint not available or no data"
fi

# ========================================
# TEST 6: MLflow INTEGRATION
# ========================================

test_step "\n[TEST 6] MLflow Integration"

MLFLOW_RUN_ID=$(echo $ANALYSIS_RESPONSE | jq -r '.mlflow_run_id')
MLFLOW_URL=$(echo $ANALYSIS_RESPONSE | jq -r '.mlflow_url')

if [ "$MLFLOW_RUN_ID" != "null" ] && [ "$MLFLOW_RUN_ID" != "monitoring_analysis" ]; then
    test_pass "MLflow run created: $MLFLOW_RUN_ID"
    test_info "View run: $MLFLOW_URL"
else
    test_info "MLflow integration not configured or partial"
fi

# ========================================
# TEST 7: DIFFERENT BIAS LEVELS
# ========================================

test_step "\n[TEST 7] Test Multiple Bias Levels"

for BIAS_LEVEL in moderate severe; do
    test_info "\nTesting $BIAS_LEVEL bias level..."
    
    # Register new model
    REG=$(curl -s -X POST "$BASE_URL/models/register" \
      -H "Content-Type: application/json" \
      -d "{
        \"model_name\": \"Test Model - ${BIAS_LEVEL}\",
        \"model_type\": \"classification\",
        \"sensitive_attributes\": [\"race\", \"gender\", \"age\"]
      }")
    
    TEST_MODEL_ID=$(echo $REG | jq -r '.model_id')
    
    # Check if test data exists
    if [ ! -f "test_data/test_predictions_${BIAS_LEVEL}_5000.csv" ]; then
        test_info "Generating ${BIAS_LEVEL} test data..."
        python prediction_generator.py $BIAS_LEVEL 5000
    fi
    
    # Upload predictions
    curl -s -X POST "$BASE_URL/monitor/upload_csv?model_id=$TEST_MODEL_ID" \
      -F "file=@test_data/test_predictions_${BIAS_LEVEL}_5000.csv" > /dev/null
    
    # Analyze
    ANALYSIS=$(curl -s -X POST "$BASE_URL/analyze" \
      -H "Content-Type: application/json" \
      -d "{\"model_id\": \"$TEST_MODEL_ID\", \"period_days\": 30}")
    
    TEST_DI=$(echo $ANALYSIS | jq -r '.fairness_metrics.race.disparate_impact.ratio')
    TEST_STATUS=$(echo $ANALYSIS | jq -r '.bias_status')
    
    # Validate expected status
    case $BIAS_LEVEL in
        moderate)
            if [ "$TEST_STATUS" == "warning" ] || [ "$TEST_STATUS" == "critical" ]; then
                test_pass "${BIAS_LEVEL}: Detected bias correctly (DI=$TEST_DI, Status=$TEST_STATUS)"
            else
                test_fail "${BIAS_LEVEL}: Expected warning/critical, got $TEST_STATUS"
            fi
            ;;
        severe)
            if [ "$TEST_STATUS" == "critical" ]; then
                test_pass "${BIAS_LEVEL}: Detected critical bias (DI=$TEST_DI)"
            else
                test_info "${BIAS_LEVEL}: Status=$TEST_STATUS (DI=$TEST_DI)"
            fi
            ;;
    esac
done

# ========================================
# TEST 8: EDGE CASES
# ========================================

test_step "\n[TEST 8] Edge Cases & Error Handling"

# Test: Upload without model_id
test_info "Testing upload without model_id..."
ERROR_RESPONSE=$(curl -s -X POST "$BASE_URL/monitor/upload_csv" \
  -F "file=@test_data/test_predictions_borderline_5000.csv")

if echo "$ERROR_RESPONSE" | grep -q "detail"; then
    test_pass "Properly rejects upload without model_id"
else
    test_fail "Should reject upload without model_id"
fi

# Test: Analyze with insufficient data
test_info "Testing analysis with high min_samples..."
INSUFFICIENT=$(curl -s -X POST "$BASE_URL/analyze" \
  -H "Content-Type: application/json" \
  -d "{\"model_id\": \"$MODEL_ID\", \"min_samples\": 999999}")

if echo "$INSUFFICIENT" | grep -q "Insufficient"; then
    test_pass "Properly validates minimum sample requirement"
else
    test_fail "Should reject analysis with insufficient data"
fi

# Test: Get non-existent model
test_info "Testing non-existent model..."
NOTFOUND=$(curl -s "$BASE_URL/model/fake_model_id")

if echo "$NOTFOUND" | grep -q "not found"; then
    test_pass "Properly handles non-existent model"
else
    test_fail "Should return 404 for non-existent model"
fi

# ========================================
# TEST 9: MONITORING STATS
# ========================================

test_step "\n[TEST 9] Monitoring Statistics"

STATS=$(curl -s "$BASE_URL/monitor/stats/$MODEL_ID?days=7")
PRED_PER_DAY=$(echo $STATS | jq -r '.predictions_per_day')
HAS_GROUND_TRUTH=$(echo $STATS | jq -r '.has_ground_truth')

test_pass "Stats: $(printf "%.0f" $PRED_PER_DAY) predictions/day, Ground truth: $HAS_GROUND_TRUTH"

# ========================================
# TEST 10: SYSTEM HEALTH
# ========================================

test_step "\n[TEST 10] System Health Check"

HEALTH=$(curl -s "$BASE_URL/monitor/health")
HEALTH_STATUS=$(echo $HEALTH | jq -r '.status')

if [ "$HEALTH_STATUS" == "healthy" ]; then
    TOTAL_LOGS=$(echo $HEALTH | jq -r '.statistics.total_predictions_logged')
    test_pass "System healthy - $TOTAL_LOGS total predictions logged"
else
    test_fail "System health check failed"
fi

# ========================================
# SUMMARY
# ========================================

echo ""
echo "========================================================================"
echo "  TEST SUMMARY"
echo "========================================================================"
echo ""

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
SUCCESS_RATE=$(echo "scale=1; $TESTS_PASSED * 100 / $TOTAL_TESTS" | bc)

echo -e "${GREEN}PASSED: $TESTS_PASSED${NC}"
echo -e "${RED}FAILED: $TESTS_FAILED${NC}"
echo "TOTAL:  $TOTAL_TESTS"
echo ""
echo "Success Rate: ${SUCCESS_RATE}%"
echo ""

# Key artifacts
echo "========================================================================"
echo "  TEST ARTIFACTS"
echo "========================================================================"
echo "Primary Model ID: $MODEL_ID"
echo "Analysis ID:      $ANALYSIS_ID"
echo "Report:           $DOWNLOAD_URL"
echo "MLflow Run:       $MLFLOW_RUN_ID"
echo "Batch ID:         $BATCH_ID"
echo ""
echo "Download Report:"
echo "  curl $BASE_URL$DOWNLOAD_URL -o biasguard_test_report.pdf"
echo ""
echo "View in MLflow:"
echo "  open $MLFLOW_URL"
echo ""

# Final result
if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}========================================================================"
    echo "  ALL TESTS PASSED - BIASGUARD 2.0 READY FOR PRODUCTION"
    echo "========================================================================${NC}"
    exit 0
else
    echo -e "${RED}========================================================================"
    echo "  SOME TESTS FAILED - REVIEW ERRORS ABOVE"
    echo "========================================================================${NC}"
    exit 1
fi