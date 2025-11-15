from core.src.exception import CustomException
import os, sys
import numpy as np
import pandas as pd


def preprocess_inputs(y_true, y_pred, sensitive_attr):
    """Clean and validate inputs before metric calculation."""
    try:
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        sensitive_attr = np.array(sensitive_attr)
        
        # Handle categorical sensitive attributes (Male/Female → 0/1)
        if sensitive_attr.dtype.kind in 'OSU':  # Object, string, unicode
            unique_vals = np.unique(sensitive_attr)
            if len(unique_vals) == 2:
                sensitive_attr = (sensitive_attr == unique_vals[1]).astype(int)
        
        # Handle multi-class sensitive attributes (Age: 18,19,20... → binary split)
        unique_vals = np.unique(sensitive_attr)
        if len(unique_vals) > 2:
            if pd.api.types.is_numeric_dtype(sensitive_attr):
                median_val = np.median(sensitive_attr)
                sensitive_attr = (sensitive_attr >= median_val).astype(int)
            else:
                # Most frequent vs others
                most_frequent = pd.Series(sensitive_attr).mode()[0]
                sensitive_attr = (sensitive_attr == most_frequent).astype(int)
        
        # Basic validation
        if len(y_true) != len(y_pred) or len(y_true) != len(sensitive_attr):
            raise ValueError("All input arrays must have the same length")
        
        # Check group sizes
        unique_groups, counts = np.unique(sensitive_attr, return_counts=True)
        if len(unique_groups) < 2:
            raise ValueError("Need at least 2 groups in sensitive attribute")
        if np.min(counts) < 5:
            print(f"Warning: Small group size detected (min: {np.min(counts)} samples)")
        
        return y_true, y_pred, sensitive_attr
        
    except Exception as e:
        raise CustomException(e, sys)


def statistical_parity(y_true, y_pred, sensitive_attr, threshold=0.1):
    '''Calculates Parity Difference : Diff between Two Sensitive Parity Groups'''
    
    try:
        # Enhanced preprocessing
        _, y_pred, sensitive_attr = preprocess_inputs(y_true, y_pred, sensitive_attr)
        
        group_0_pred = y_pred[sensitive_attr == 0]
        group_0_rate = np.mean(group_0_pred)
        
        group_1_pred = y_pred[sensitive_attr == 1]
        group_1_rate = np.mean(group_1_pred)
        
        diff = abs(group_0_rate - group_1_rate)
        bias = diff > threshold
        
        # Enhanced output format
        return {
            'metric': 'Statistical Parity Difference',
            'group_0_rate': round(group_0_rate, 4),
            'group_1_rate': round(group_1_rate, 4),
            'statistical_parity_diff': round(diff, 4),
            'bias_detected': bias,
            'threshold': threshold,
            'severity': 'HIGH' if diff > threshold * 2 else 'MODERATE' if bias else 'LOW',
            'interpretation': f"Groups differ in positive prediction rates by {diff:.3f}"
        }
        
    except Exception as e:
        raise CustomException(e, sys)


def equal_oppurtunity_check(y_true, y_pred, sensitive_attr):
    try:
        # Enhanced preprocessing
        y_true, y_pred, sensitive_attr = preprocess_inputs(y_true, y_pred, sensitive_attr)

        positive_label = 1
        groups = np.unique(sensitive_attr)

        tpr = []
        for group in groups:
            mask = (sensitive_attr == group)
            tp = np.sum((y_true[mask] == positive_label) & (y_pred[mask] == positive_label))
            pos = np.sum(y_true[mask] == positive_label)
            tpr.append(tp / pos if pos > 0 else 0)

        difference = abs(tpr[0] - tpr[1])
        
        # Enhanced output format
        return {
            'metric': 'Equal Opportunity',
            'group_0_tpr': round(tpr[0], 4),
            'group_1_tpr': round(tpr[1], 4) if len(tpr) > 1 else 0,
            'difference': round(difference, 4),
            'bias_detected': difference > 0.1,
            'threshold': 0.1,
            'severity': 'HIGH' if difference > 0.2 else 'MODERATE' if difference > 0.1 else 'LOW',
            'interpretation': f"Groups differ in true positive rates by {difference:.3f}"
        }
        
    except Exception as e:
        raise CustomException(e, sys)


def disparate_impact_ratio(y_true, y_pred, sensitive_attr):
    """
    Ratio of favorable outcomes between groups.
    """
    try:
        # Enhanced preprocessing
        _, y_pred, sensitive_attr = preprocess_inputs(y_true, y_pred, sensitive_attr)
        
        favorable = 1
        groups = np.unique(sensitive_attr)

        rates = []
        for group in groups:
            mask = (sensitive_attr == group)
            favorable_rate = np.sum(y_pred[mask] == favorable) / np.sum(mask)
            rates.append(favorable_rate)

        ratio = rates[0] / rates[1] if rates[1] != 0 else np.inf
        
        # Enhanced output format
        return {
            'metric': 'Disparate Impact Ratio',
            'group_0_rate': round(rates[0], 4),
            'group_1_rate': round(rates[1], 4) if len(rates) > 1 else 0,
            'ratio': round(ratio, 4) if ratio != np.inf else 'inf',
            'bias_detected': not (0.8 <= ratio <= 1.2) if ratio != np.inf else True,
            'threshold': (0.8, 1.2),
            'severity': 'HIGH' if ratio == np.inf or ratio < 0.6 or ratio > 1.4 else 'MODERATE' if not (0.8 <= ratio <= 1.2) else 'LOW',
            'interpretation': f"Ratio of favorable outcomes between groups: {ratio:.3f}" if ratio != np.inf else "One group receives no favorable outcomes"
        }
        
    except Exception as e:
        raise CustomException(e, sys)


def average_odds_difference(y_true, y_pred, sensitive_attr):
    """
    Difference in average of TPR and FPR between groups.
    """
    try:
        # Enhanced preprocessing
        y_true, y_pred, sensitive_attr = preprocess_inputs(y_true, y_pred, sensitive_attr)
        
        groups = np.unique(sensitive_attr)
        odds = []

        for group in groups:
            mask = (sensitive_attr == group)

            tp = np.sum((y_true[mask] == 1) & (y_pred[mask] == 1))
            fn = np.sum((y_true[mask] == 1) & (y_pred[mask] == 0))
            fp = np.sum((y_true[mask] == 0) & (y_pred[mask] == 1))
            tn = np.sum((y_true[mask] == 0) & (y_pred[mask] == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            odds.append((tpr, fpr))

        avg_diff = abs((odds[0][0] - odds[1][0]) + (odds[0][1] - odds[1][1])) / 2
        
        # Enhanced output format
        return {
            'metric': 'Average Odds Difference',
            'group_0_tpr': round(odds[0][0], 4),
            'group_0_fpr': round(odds[0][1], 4),
            'group_1_tpr': round(odds[1][0], 4),
            'group_1_fpr': round(odds[1][1], 4),
            'average_difference': round(avg_diff, 4),
            'bias_detected': avg_diff > 0.1,
            'threshold': 0.1,
            'severity': 'HIGH' if avg_diff > 0.2 else 'MODERATE' if avg_diff > 0.1 else 'LOW',
            'interpretation': f"Average of TPR and FPR differences: {avg_diff:.3f}"
        }
        
    except Exception as e:
        raise CustomException(e, sys)


def false_positive_rate_parity(y_true, y_pred, sensitive_attr):
    """
    NEW METRIC 1: Calculates difference in False Positive Rates between groups.
    """
    try:
        # Enhanced preprocessing
        y_true, y_pred, sensitive_attr = preprocess_inputs(y_true, y_pred, sensitive_attr)
        
        groups = np.unique(sensitive_attr)
        fpr_rates = []
        
        for group in groups:
            mask = (sensitive_attr == group)
            
            fp = np.sum((y_true[mask] == 0) & (y_pred[mask] == 1))
            tn = np.sum((y_true[mask] == 0) & (y_pred[mask] == 0))
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fpr_rates.append(fpr)
        
        fpr_difference = abs(fpr_rates[0] - fpr_rates[1]) if len(fpr_rates) >= 2 else 0
        
        return {
            'metric': 'False Positive Rate Parity',
            'group_0_fpr': round(fpr_rates[0], 4),
            'group_1_fpr': round(fpr_rates[1], 4) if len(fpr_rates) > 1 else 0,
            'fpr_difference': round(fpr_difference, 4),
            'bias_detected': fpr_difference > 0.1,
            'threshold': 0.1,
            'severity': 'HIGH' if fpr_difference > 0.2 else 'MODERATE' if fpr_difference > 0.1 else 'LOW',
            'interpretation': f"Groups differ in false positive rates by {fpr_difference:.3f}"
        }
        
    except Exception as e:
        raise CustomException(e, sys)


def predictive_parity(y_true, y_pred, sensitive_attr):
    """
    NEW METRIC 2: Calculates difference in Positive Predictive Value (Precision) between groups.
    """
    try:
        # Enhanced preprocessing
        y_true, y_pred, sensitive_attr = preprocess_inputs(y_true, y_pred, sensitive_attr)
        
        groups = np.unique(sensitive_attr)
        ppv_rates = []
        
        for group in groups:
            mask = (sensitive_attr == group)
            
            tp = np.sum((y_true[mask] == 1) & (y_pred[mask] == 1))
            fp = np.sum((y_true[mask] == 0) & (y_pred[mask] == 1))
            
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            ppv_rates.append(ppv)
        
        ppv_difference = abs(ppv_rates[0] - ppv_rates[1]) if len(ppv_rates) >= 2 else 0
        
        return {
            'metric': 'Predictive Parity',
            'group_0_ppv': round(ppv_rates[0], 4),
            'group_1_ppv': round(ppv_rates[1], 4) if len(ppv_rates) > 1 else 0,
            'ppv_difference': round(ppv_difference, 4),
            'bias_detected': ppv_difference > 0.1,
            'threshold': 0.1,
            'severity': 'HIGH' if ppv_difference > 0.2 else 'MODERATE' if ppv_difference > 0.1 else 'LOW',
            'interpretation': f"Groups differ in positive predictive value by {ppv_difference:.3f}"
        }
        
    except Exception as e:
        raise CustomException(e, sys)


def treatment_equality(y_true, y_pred, sensitive_attr):
    """
    NEW METRIC 4: Calculates ratio of False Negatives to False Positives for each group.
    """
    try:
        # Enhanced preprocessing
        y_true, y_pred, sensitive_attr = preprocess_inputs(y_true, y_pred, sensitive_attr)
        
        groups = np.unique(sensitive_attr)
        fn_fp_ratios = []
        
        for group in groups:
            mask = (sensitive_attr == group)
            
            fn = np.sum((y_true[mask] == 1) & (y_pred[mask] == 0))
            fp = np.sum((y_true[mask] == 0) & (y_pred[mask] == 1))
            
            # Handle zero division safely
            if fp == 0:
                ratio = float('inf') if fn > 0 else 1.0
            else:
                ratio = fn / fp
                
            fn_fp_ratios.append(ratio)
        
        # Calculate difference
        if len(fn_fp_ratios) >= 2:
            if np.inf in fn_fp_ratios:
                ratio_difference = float('inf')
            else:
                ratio_difference = abs(fn_fp_ratios[0] - fn_fp_ratios[1])
        else:
            ratio_difference = 0
        
        return {
            'metric': 'Treatment Equality',
            'group_0_fn_fp_ratio': round(fn_fp_ratios[0], 4) if fn_fp_ratios[0] != float('inf') else 'inf',
            'group_1_fn_fp_ratio': round(fn_fp_ratios[1], 4) if len(fn_fp_ratios) > 1 and fn_fp_ratios[1] != float('inf') else 'inf',
            'ratio_difference': round(ratio_difference, 4) if ratio_difference != float('inf') else 'inf',
            'bias_detected': ratio_difference > 1.0 if ratio_difference != float('inf') else True,
            'threshold': 1.0,
            'severity': 'HIGH' if ratio_difference == float('inf') or ratio_difference > 2.0 else 'MODERATE' if ratio_difference > 1.0 else 'LOW',
            'interpretation': f"Groups differ in FN/FP cost ratio by {ratio_difference}" if ratio_difference != float('inf') else "Infinite difference - one group has no false positives"
        }
        
    except Exception as e:
        raise CustomException(e, sys)