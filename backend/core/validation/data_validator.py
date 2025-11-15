import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from core.src.logger import logging

class DataValidator:
    
    def __init__(self, 
                 min_samples: int = 500, #was 1000 check the new notes file for parameters precision
                 min_group_size: int = 25, #50
                 max_missing_pct: float = 20.0, #10
                 max_class_imbalance: float = 0.02, #0.05
                 max_correlation_threshold: float = 0.98): #0.95
        self.min_samples = min_samples
        self.min_group_size = min_group_size
        self.max_missing_pct = max_missing_pct
        self.max_class_imbalance = max_class_imbalance
        self.max_correlation_threshold = max_correlation_threshold
    
    def validate_dataset(self, 
                        df: pd.DataFrame, 
                        target_col: str, 
                        sensitive_cols: List[str]) -> Dict:
        
        blockers = []
        warnings = []
        metadata = {}
        
        if len(df) < self.min_samples:
            if len(df) < 500:
                blockers.append(f"BLOCKER: Dataset too small ({len(df)} rows). Need at least 500 samples for bias detection.")
            else:
                warnings.append(f"WARNING: Dataset has {len(df)} rows. Recommended minimum: {self.min_samples} for reliable metrics.")
        
        metadata["total_samples"] = len(df)
        
        if target_col not in df.columns:
            blockers.append(f"BLOCKER: Target column '{target_col}' not found in dataset.")
            return {"valid": False, "blockers": blockers, "warnings": warnings, "metadata": metadata}
        
        target_dist = df[target_col].value_counts(normalize=True)
        min_class_pct = target_dist.min()
        
        if min_class_pct < self.max_class_imbalance:
            blockers.append(f"BLOCKER: Severe class imbalance - minority class is {min_class_pct:.1%} of data (need >{self.max_class_imbalance:.0%})")
        elif min_class_pct < 0.10:
            warnings.append(f"WARNING: Class imbalance detected - minority class is {min_class_pct:.1%}")
        
        metadata["class_distribution"] = target_dist.to_dict()
        
        for col in sensitive_cols:
            if col not in df.columns:
                blockers.append(f"BLOCKER: Sensitive column '{col}' not found in dataset.")
                continue
            
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            if missing_pct > self.max_missing_pct:
                warnings.append(f"WARNING: '{col}' has {missing_pct:.1f}% missing values (threshold: {self.max_missing_pct}%)")
            
            unique_count = df[col].nunique()
            is_continuous = pd.api.types.is_numeric_dtype(df[col]) and unique_count > 20
            
            if is_continuous:
                total_valid = df[col].notna().sum()
                if total_valid < self.min_group_size:
                    blockers.append(f"BLOCKER: '{col}' has insufficient valid samples ({total_valid})")
                else:
                    warnings.append(f"INFO: '{col}' appears continuous ({unique_count} unique values) - will be binned for bias analysis")
                
                metadata[f"{col}_type"] = "continuous"
                metadata[f"{col}_range"] = [float(df[col].min()), float(df[col].max())]
            else:
                group_sizes = df[col].value_counts()
                min_group = group_sizes.min()
                
                if min_group < self.min_group_size:
                    blockers.append(f"BLOCKER: '{col}' has a group with only {min_group} samples (need >{self.min_group_size} per group for reliable metrics)")
                elif min_group < 100:
                    warnings.append(f"WARNING: '{col}' has a small group ({min_group} samples). Metrics may be unstable.")
                
                metadata[f"{col}_group_sizes"] = group_sizes.to_dict()
        
        feature_cols = [c for c in df.columns if c not in [target_col] + sensitive_cols]
        
        leakage_features = []
        for feat in feature_cols:
            if pd.api.types.is_numeric_dtype(df[feat]) and pd.api.types.is_numeric_dtype(df[target_col]):
                corr = abs(df[feat].corr(df[target_col]))
                if corr > self.max_correlation_threshold:
                    leakage_features.append((feat, corr))
        
        if leakage_features:
            for feat, corr in leakage_features:
                warnings.append(f"WARNING: Feature '{feat}' has {corr:.2f} correlation with target (possible label leakage)")
        
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            dup_pct = (duplicates / len(df)) * 100
            if dup_pct > 5:
                warnings.append(f"WARNING: {duplicates} duplicate rows ({dup_pct:.1f}%)")
        
        metadata["duplicate_rows"] = int(duplicates)
        
        is_valid = len(blockers) == 0
        
        return {
            "valid": is_valid,
            "blockers": blockers,
            "warnings": warnings,
            "metadata": metadata
        }