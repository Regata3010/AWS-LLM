"""
backend/core/bias_detector/intersectionality.py
BULLETPROOF VERSION - NO VARIABLE COLLISION
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from itertools import combinations
from core.src.logger import logging

class IntersectionalityAnalyzer:
    
    def __init__(self, min_group_size=30, di_threshold=0.8, max_combinations=20):
        self.min_group_size = min_group_size
        self.di_threshold = di_threshold
        self.max_combinations = max_combinations
    
    def analyze(self, y_test_array, y_pred_array, sensitive_dict):
        """
        Analyze intersectional bias
        
        Args:
            y_test_array: True labels (numpy array)
            y_pred_array: Predictions (numpy array)
            sensitive_dict: Dict of sensitive attributes
        
        Returns:
            Complete intersectionality results dict
        """
        
        # PRE-INITIALIZE ALL FIELDS
        analysis_output = {
            "intersectional_groups": {},
            "worst_disparities": [],
            "best_outcomes": [],
            "summary": {
                "total_groups_analyzed": 0,
                "groups_below_threshold": 0,
                "groups_above_threshold": 0,
                "max_disparity": 0.0,
                "min_disparity": 999.0
            },
            "recommendations": []
        }
        
        try:
            # Validate inputs
            if len(sensitive_dict) < 2:
                analysis_output["recommendations"] = ["Need at least 2 sensitive attributes for intersectionality"]
                return analysis_output
            
            if len(y_test_array) == 0 or len(y_pred_array) == 0:
                analysis_output["recommendations"] = ["Empty prediction arrays"]
                return analysis_output
            
            logging.info(f"Starting intersectionality: {len(sensitive_dict)} attributes, {len(y_pred_array)} samples")
            
            # Calculate baseline
            overall_positive_rate = float(np.mean(y_pred_array))
            
            if overall_positive_rate == 0:
                analysis_output["recommendations"] = ["No positive predictions - cannot calculate intersectionality"]
                return analysis_output
            
            # Get attribute names
            attr_list = list(sensitive_dict.keys())
            
            # Analyze all pairs
            for attr_a, attr_b in combinations(attr_list, 2):
                
                vals_a = sensitive_dict[attr_a]
                vals_b = sensitive_dict[attr_b]
                
                unique_a = np.unique(vals_a)
                unique_b = np.unique(vals_b)
                
                for val_a in unique_a:
                    for val_b in unique_b:
                        
                        # Create mask for this group
                        group_mask = (vals_a == val_a) & (vals_b == val_b)
                        group_count = int(np.sum(group_mask))
                        
                        # Skip small groups
                        if group_count < self.min_group_size:
                            continue
                        
                        # Count analyzed groups
                        analysis_output["summary"]["total_groups_analyzed"] += 1
                        
                        # Calculate metrics
                        group_predictions = y_pred_array[group_mask]
                        group_positive_rate = float(np.mean(group_predictions))
                        
                        # Disparate impact
                        di_score = group_positive_rate / overall_positive_rate
                        
                        # Risk level
                        if di_score < 0.5:
                            risk = "CRITICAL"
                        elif di_score < self.di_threshold:
                            risk = "HIGH"
                        elif di_score < 0.85:
                            risk = "MODERATE"
                        elif di_score > 1.25:
                            risk = "MODERATE"
                        else:
                            risk = "LOW"
                        
                        # Build group name
                        group_id = f"{attr_a}={val_a}, {attr_b}={val_b}"
                        
                        # Store results
                        analysis_output["intersectional_groups"][group_id] = {
                            "sample_size": group_count,
                            "positive_rate": round(group_positive_rate, 4),
                            "disparate_impact": round(di_score, 4),
                            "risk_level": risk,
                            "compliant": (self.di_threshold <= di_score <= 1.25)
                        }
                        
                        # Update summary
                        if di_score < self.di_threshold:
                            analysis_output["summary"]["groups_below_threshold"] += 1
                        elif di_score > 1.25:
                            analysis_output["summary"]["groups_above_threshold"] += 1
                        
                        if di_score < analysis_output["summary"]["min_disparity"]:
                            analysis_output["summary"]["min_disparity"] = di_score
                        
                        if di_score > analysis_output["summary"]["max_disparity"]:
                            analysis_output["summary"]["max_disparity"] = di_score
            
            # Find worst groups
            if len(analysis_output["intersectional_groups"]) > 0:
                sorted_by_di = sorted(
                    analysis_output["intersectional_groups"].items(),
                    key=lambda x: x[1]["disparate_impact"]
                )
                
                for grp_name, grp_data in sorted_by_di[:5]:
                    if grp_data["disparate_impact"] < self.di_threshold:
                        analysis_output["worst_disparities"].append({
                            "group": grp_name,
                            "disparate_impact": grp_data["disparate_impact"],
                            "positive_rate": grp_data["positive_rate"],
                            "sample_size": grp_data["sample_size"],
                            "severity": grp_data["risk_level"]
                        })
            
            # Generate recommendations (USE LOCAL VARIABLE)
            my_recs = []
            
            total_analyzed = analysis_output["summary"]["total_groups_analyzed"]
            groups_at_risk = analysis_output["summary"]["groups_below_threshold"]
            
            if total_analyzed == 0:
                my_recs.append("No intersectional groups with sufficient sample size")
            elif groups_at_risk == 0:
                my_recs.append("✅ No intersectional bias detected")
            else:
                my_recs.append(f"⚠️  {groups_at_risk}/{total_analyzed} intersectional groups below threshold")
                
                if len(analysis_output["worst_disparities"]) > 0:
                    worst = analysis_output["worst_disparities"][0]
                    my_recs.append(f"Worst group: {worst['group']} (DI={worst['disparate_impact']:.2f})")
            
            analysis_output["recommendations"] = my_recs
            
            logging.info(f"Intersectionality complete: {total_analyzed} groups analyzed")
            
            return analysis_output
            
        except Exception as err:
            logging.error(f"Intersectionality crashed: {err}", exc_info=True)
            analysis_output["recommendations"] = [f"Analysis error: {str(err)}"]
            return analysis_output