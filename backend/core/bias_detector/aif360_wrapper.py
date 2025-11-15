from core.src.logger import logging
from core.src.exception import CustomException
import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
import sys


def audit_with_aif360(df: pd.DataFrame, target_col: str, sensitive_col: str):
    try:
        df = df.copy()
        df = df[[sensitive_col, target_col]].dropna()
        
        
        unique_vals = df[sensitive_col].unique()
        logging.info(f"Sensitive column '{sensitive_col}' has {len(unique_vals)} unique values")
        
        # If sensitive attribute has more than 2 values, bin it
        if len(unique_vals) > 2:
            if pd.api.types.is_numeric_dtype(df[sensitive_col]):
                # For numeric data (like age), bin around median
                median_val = df[sensitive_col].median()
                df[sensitive_col] = (df[sensitive_col] >= median_val).astype(int)
                logging.info(f"Binned numeric sensitive attribute around median {median_val}")
            else:
                # For categorical data, use most frequent vs others
                most_frequent = df[sensitive_col].mode()[0]
                df[sensitive_col] = (df[sensitive_col] == most_frequent).astype(int)
                logging.info(f"Binned categorical sensitive attribute: '{most_frequent}' vs others")
        
        elif df[sensitive_col].dtype == 'object' or df[sensitive_col].dtype.name == 'category':
            unique_vals = df[sensitive_col].dropna().unique()
            if len(unique_vals) != 2:
                raise ValueError(f"Sensitive attribute '{sensitive_col}' must be binary. Found values: {unique_vals}")
            df[sensitive_col] = df[sensitive_col].map({unique_vals[0]: 0, unique_vals[1]: 1})

        # Handle target attribute (your existing code)
        if df[target_col].dtype == 'object' or df[target_col].dtype.name == 'category':
            unique_vals = df[target_col].dropna().unique()
            if len(unique_vals) != 2:
                raise ValueError(f"Target attribute '{target_col}' must be binary. Found values: {unique_vals}")
            df[target_col] = df[target_col].map({unique_vals[0]: 0, unique_vals[1]: 1})

        
        df[sensitive_col] = df[sensitive_col].astype(int)
        df[target_col] = df[target_col].astype(int)

        # Verify binary (your existing code)
        if set(df[sensitive_col].unique()) != {0, 1}:
            raise ValueError(f"Sensitive attribute must be binary (0,1). Found: {df[sensitive_col].unique()}")
        if set(df[target_col].unique()) != {0, 1}:
            raise ValueError(f"Target attribute must be binary (0,1). Found: {df[target_col].unique()}")

        dataset = BinaryLabelDataset(
            df=df,
            label_names=[target_col],
            protected_attribute_names=[sensitive_col]
        )

        fairness_metrics = BinaryLabelDatasetMetric(
            dataset,
            unprivileged_groups=[{sensitive_col: 0}],
            privileged_groups=[{sensitive_col: 1}]
        )

        mean_diff = fairness_metrics.mean_difference()
        disparate_impact = fairness_metrics.disparate_impact()
        consistency = fairness_metrics.consistency()
        
        # Convert numpy arrays to floats (your existing code)
        if isinstance(mean_diff, np.ndarray):
            mean_diff = float(mean_diff.item()) if mean_diff.size == 1 else float(mean_diff[0])
        if isinstance(disparate_impact, np.ndarray):
            disparate_impact = float(disparate_impact.item()) if disparate_impact.size == 1 else float(disparate_impact[0])
        if isinstance(consistency, np.ndarray):
            consistency = float(consistency.item()) if consistency.size == 1 else float(consistency[0])

        results = {
            "Mean Difference": round(float(mean_diff), 4),
            "Disparate Impact": round(float(disparate_impact), 4),
            "Consistency": round(float(consistency), 4)
        }

        return results

    except Exception as e:
        raise CustomException(e, sys)



if __name__ == "__main__":
    df = pd.read_csv("dfs/Adult.csv")
    
    target_col = "income"
    sensitive_col = "age"  
    
    try:
        report = audit_with_aif360(df, target_col, sensitive_col)
        print("AIF360 Audit Report:")
        for k, v in report.items():
            print(f"{k}: {v}")
    except Exception as e:
        print("Error in audit:")
        print(e)