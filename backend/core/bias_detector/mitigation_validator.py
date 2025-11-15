"""
backend/core/bias_detector/mitigation_validator.py

Validate that bias mitigation actually achieved its goals
Critical for ensuring mitigation strategies are effective
"""
from typing import Dict, List, Optional
from core.src.logger import logging

class MitigationValidator:
    """
    Validates mitigation effectiveness after applying bias reduction strategies
    
    Checks:
    1. Did disparate impact improve?
    2. Is accuracy loss acceptable?
    3. Did we introduce new bias elsewhere?
    4. Are we now compliant with fairness thresholds?
    """
    
    def __init__(self, 
                 min_di_threshold: float = 0.8,
                 max_di_threshold: float = 1.25,
                 max_accuracy_loss: float = 0.10,
                 max_new_bias_increase: float = 1.5):
        """
        Initialize mitigation validator with thresholds
        
        Args:
            min_di_threshold: Minimum acceptable disparate impact (CFPB: 0.8)
            max_di_threshold: Maximum acceptable disparate impact (1.25)
            max_accuracy_loss: Maximum acceptable accuracy degradation (10%)
            max_new_bias_increase: Max allowed increase in other metrics (1.5x)
        """
        self.min_di_threshold = min_di_threshold
        self.max_di_threshold = max_di_threshold
        self.max_accuracy_loss = max_accuracy_loss
        self.max_new_bias_increase = max_new_bias_increase
    
    def validate(self, 
                original_metrics: Dict, 
                new_metrics: Dict, 
                strategy: str) -> Dict:
        """
        Comprehensive validation of mitigation success
        
        Args:
            original_metrics: Metrics before mitigation
            new_metrics: Metrics after mitigation
            strategy: Strategy used (reweighing, threshold_optimization, etc.)
            
        Returns:
            {
                "success": bool,
                "score": float (0-100),
                "issues": List[str],
                "warnings": List[str],
                "recommendations": List[str],
                "detailed_checks": Dict
            }
        """
        issues = []
        warnings = []
        recommendations = []
        scores = []
        detailed_checks = {}
        
        logging.info(f"Validating mitigation strategy: {strategy}")
        
        # ============================================
        # CHECK 1: Disparate Impact Improvement
        # ============================================
        di_check = self._check_disparate_impact(original_metrics, new_metrics)
        scores.append(di_check['score'])
        detailed_checks['disparate_impact'] = di_check
        
        if di_check['issues']:
            issues.extend(di_check['issues'])
        if di_check['warnings']:
            warnings.extend(di_check['warnings'])
        if di_check['recommendations']:
            recommendations.extend(di_check['recommendations'])
        
        # ============================================
        # CHECK 2: Accuracy Impact
        # ============================================
        accuracy_check = self._check_accuracy_impact(original_metrics, new_metrics)
        scores.append(accuracy_check['score'])
        detailed_checks['accuracy'] = accuracy_check
        
        if accuracy_check['issues']:
            issues.extend(accuracy_check['issues'])
        if accuracy_check['warnings']:
            warnings.extend(accuracy_check['warnings'])
        if accuracy_check['recommendations']:
            recommendations.extend(accuracy_check['recommendations'])
        
        # ============================================
        # CHECK 3: New Bias Introduction
        # ============================================
        new_bias_check = self._check_new_bias(original_metrics, new_metrics)
        scores.append(new_bias_check['score'])
        detailed_checks['new_bias'] = new_bias_check
        
        if new_bias_check['issues']:
            issues.extend(new_bias_check['issues'])
        if new_bias_check['warnings']:
            warnings.extend(new_bias_check['warnings'])
        
        # ============================================
        # CHECK 4: Compliance Status
        # ============================================
        compliance_check = self._check_compliance(new_metrics)
        scores.append(compliance_check['score'])
        detailed_checks['compliance'] = compliance_check
        
        if compliance_check['issues']:
            issues.extend(compliance_check['issues'])
        if compliance_check['warnings']:
            warnings.extend(compliance_check['warnings'])
        
        # ============================================
        # CALCULATE OVERALL SCORE
        # ============================================
        overall_score = sum(scores) / len(scores) if scores else 0
        
        # Determine success (no critical issues + score >= 60)
        success = len(issues) == 0 and overall_score >= 60
        
        # ============================================
        # GENERATE FINAL RECOMMENDATIONS
        # ============================================
        if success:
            recommendations.insert(0, "✅ Mitigation successful - model ready for deployment")
        else:
            if not recommendations:
                recommendations.append("❌ Mitigation failed - consider:")
                recommendations.append(f"  • Try different strategy (current: {strategy})")
                recommendations.append("  • Adjust mitigation parameters")
                recommendations.append("  • Review data quality issues")
        
        # Add strategy-specific recommendations
        if strategy == "threshold_optimization" and overall_score < 70:
            recommendations.append("💡 Threshold optimization may be insufficient - try reweighing or fairness constraints")
        
        if strategy in ["reweighing", "fairness_constraints"] and accuracy_check['score'] < 50:
            recommendations.append("💡 Consider threshold optimization to preserve accuracy")
        
        logging.info(f"Mitigation validation score: {overall_score:.1f}/100")
        logging.info(f"Success: {success}, Issues: {len(issues)}, Warnings: {len(warnings)}")
        
        return {
            "success": success,
            "score": round(overall_score, 1),
            "issues": issues[:5],  # Limit to top 5
            "warnings": warnings[:5],
            "recommendations": recommendations[:5],
            "detailed_checks": detailed_checks
        }
    
    def _check_disparate_impact(self, original_metrics: Dict, new_metrics: Dict) -> Dict:
        """Check if disparate impact improved"""
        orig_di = self._safe_float(original_metrics.get("disparate_impact", 1))
        new_di = self._safe_float(new_metrics.get("disparate_impact", 1))
        
        issues = []
        warnings = []
        recommendations = []
        score = 0
        
        if orig_di is None or new_di is None:
            warnings.append("⚠️  Disparate impact values contain infinity - cannot validate")
            score = 50
        else:
            di_improvement = new_di - orig_di
            
            # Case 1: DI got WORSE
            if di_improvement < 0:
                issues.append(
                    f"❌ Disparate impact WORSENED: {orig_di:.2f} → {new_di:.2f} "
                    f"(change: {di_improvement:.2f})"
                )
                score = 0
                recommendations.append("Mitigation made bias worse - try different strategy")
            
            # Case 2: DI improved but still below threshold
            elif new_di < self.min_di_threshold:
                issues.append(
                    f"❌ Still below {self.min_di_threshold} threshold after mitigation "
                    f"(DI: {new_di:.2f})"
                )
                score = 30
                recommendations.append(
                    f"DI improved by {di_improvement:.2f} but insufficient - "
                    "try more aggressive mitigation"
                )
            
            # Case 3: DI improved but above max threshold
            elif new_di > self.max_di_threshold:
                warnings.append(
                    f"⚠️  Disparate impact above {self.max_di_threshold} "
                    f"(now favoring previously disadvantaged group: {new_di:.2f})"
                )
                score = 70
            
            # Case 4: SUCCESS - in acceptable range
            else:
                score = 100
                logging.info(
                    f"✅ Disparate impact improved: {orig_di:.2f} → {new_di:.2f} "
                    f"(improvement: {di_improvement:.2f})"
                )
        
        return {
            "score": score,
            "issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
            "original_di": orig_di,
            "new_di": new_di,
            "improvement": new_di - orig_di if (orig_di and new_di) else None
        }
    
    def _check_accuracy_impact(self, original_metrics: Dict, new_metrics: Dict) -> Dict:
        """Check if accuracy loss is acceptable"""
        orig_acc = original_metrics.get("accuracy", 0)
        new_acc = new_metrics.get("accuracy", 0)
        
        issues = []
        warnings = []
        recommendations = []
        score = 0
        
        acc_loss = orig_acc - new_acc
        
        # Case 1: Accuracy IMPROVED (rare but possible)
        if acc_loss < 0:
            score = 100
            logging.info(f"✅ Accuracy actually improved: {orig_acc:.3f} → {new_acc:.3f}")
        
        # Case 2: Accuracy loss SEVERE (>10%)
        elif acc_loss > self.max_accuracy_loss:
            issues.append(
                f"❌ Accuracy loss too high: {acc_loss:.1%} "
                f"(threshold: {self.max_accuracy_loss:.1%})"
            )
            issues.append(f"   Original: {orig_acc:.1%}, New: {new_acc:.1%}")
            score = 20
            recommendations.append(
                "Try threshold_optimization to preserve accuracy better"
            )
        
        # Case 3: Accuracy loss MODERATE (5-10%)
        elif acc_loss > 0.05:
            warnings.append(
                f"⚠️  Moderate accuracy loss: {acc_loss:.1%} "
                f"({orig_acc:.1%} → {new_acc:.1%})"
            )
            score = 70
        
        # Case 4: Accuracy loss MINIMAL (<5%)
        else:
            score = 100
            logging.info(f"✅ Acceptable accuracy loss: {acc_loss:.1%}")
        
        return {
            "score": score,
            "issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
            "original_accuracy": orig_acc,
            "new_accuracy": new_acc,
            "accuracy_loss": acc_loss
        }
    
    def _check_new_bias(self, original_metrics: Dict, new_metrics: Dict) -> Dict:
        """Check if mitigation introduced new bias in other metrics"""
        issues = []
        warnings = []
        score = 100  # Start optimistic
        
        # Check Statistical Parity Difference
        orig_sp = abs(original_metrics.get("statistical_parity_diff", 0))
        new_sp = abs(new_metrics.get("statistical_parity_diff", 0))
        
        if new_sp > orig_sp * self.max_new_bias_increase:
            issues.append(
                f"❌ New bias in statistical parity: {orig_sp:.3f} → {new_sp:.3f} "
                f"({(new_sp/orig_sp - 1)*100:.0f}% increase)"
            )
            score -= 30
        
        # Check Equal Opportunity Difference
        if "equal_opportunity_diff" in original_metrics and "equal_opportunity_diff" in new_metrics:
            orig_eo = abs(original_metrics.get("equal_opportunity_diff", 0))
            new_eo = abs(new_metrics.get("equal_opportunity_diff", 0))
            
            if new_eo > orig_eo * 1.3:  # Allow 30% increase
                warnings.append(
                    f"⚠️  Equal opportunity worsened: {orig_eo:.3f} → {new_eo:.3f}"
                )
                score -= 15
        
        # Check Average Odds Difference
        if "average_odds_diff" in original_metrics and "average_odds_diff" in new_metrics:
            orig_ao = abs(original_metrics.get("average_odds_diff", 0))
            new_ao = abs(new_metrics.get("average_odds_diff", 0))
            
            if new_ao > orig_ao * 1.3:
                warnings.append(
                    f"⚠️  Average odds worsened: {orig_ao:.3f} → {new_ao:.3f}"
                )
                score -= 15
        
        score = max(0, score)  # Ensure non-negative
        
        if score == 100:
            logging.info("✅ No new bias introduced in other metrics")
        
        return {
            "score": score,
            "issues": issues,
            "warnings": warnings
        }
    
    def _check_compliance(self, new_metrics: Dict) -> Dict:
        """Check if model is now compliant with fairness thresholds"""
        issues = []
        warnings = []
        score = 100
        
        new_di = self._safe_float(new_metrics.get("disparate_impact", 1))
        
        if new_di is None:
            warnings.append("⚠️  Cannot determine compliance - DI is infinity")
            score = 50
        elif new_di < self.min_di_threshold:
            issues.append(
                f"❌ Still NON-COMPLIANT: DI = {new_di:.2f} < {self.min_di_threshold}"
            )
            score = 30
        elif new_di > self.max_di_threshold:
            warnings.append(
                f"⚠️  Above upper threshold: DI = {new_di:.2f} > {self.max_di_threshold}"
            )
            score = 80
        else:
            logging.info(f"✅ COMPLIANT: DI = {new_di:.2f} in [{self.min_di_threshold}, {self.max_di_threshold}]")
            score = 100
        
        # Check statistical parity
        new_sp = abs(new_metrics.get("statistical_parity_diff", 0))
        if new_sp > 0.20:  # EEOC guideline
            warnings.append(
                f"⚠️  Statistical parity difference high: {new_sp:.3f} > 0.20"
            )
            score = min(score, 80)
        
        return {
            "score": score,
            "issues": issues,
            "warnings": warnings,
            "compliant": score >= 80
        }
    
    def _safe_float(self, value) -> Optional[float]:
        """Convert value to float, handling 'inf' strings and edge cases"""
        if value is None:
            return None
        
        if value in ['inf', 'infinity', float('inf'), float('-inf')]:
            return None
        
        if isinstance(value, str):
            if value.lower() in ['inf', 'infinity', '-inf']:
                return None
            try:
                return float(value)
            except ValueError:
                return None
        
        try:
            f = float(value)
            if f == float('inf') or f == float('-inf'):
                return None
            return f
        except (ValueError, TypeError):
            return None