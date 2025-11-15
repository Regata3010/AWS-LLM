from core.src.logger import logging
from core.src.exception import CustomException
import os, sys
import pandas as pd 
from core.bias_detector.fairnessmetric import (statistical_parity,
                                              equal_oppurtunity_check,
                                              disparate_impact_ratio,
                                              average_odds_difference,
                                              false_positive_rate_parity,
                                              predictive_parity,
                                              treatment_equality)
from core.llms.column_selector import final_columns_from_csv


def generate_report_fairness(y_true, y_pred, sensitive_attr):
    """
    Generate a comprehensive fairness report with multiple bias metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        sensitive_attr: Sensitive attribute values
        
    Returns:
        dict: Dictionary containing fairness metrics
    """
    try:
        report = {}
        
        # Calculate all fairness metrics
        report['Statistical_Parity'] = statistical_parity(y_true=y_true,y_pred=y_pred, sensitive_attr=sensitive_attr)
        report['Equal_Opportunity'] = equal_oppurtunity_check(y_true=y_true, y_pred=y_pred, sensitive_attr=sensitive_attr)
        report['Disparate_Impact_Ratio'] = disparate_impact_ratio(y_true=y_true,y_pred=y_pred, sensitive_attr=sensitive_attr)
        report['Average_Odds_Difference'] = average_odds_difference(y_true=y_true, y_pred=y_pred, sensitive_attr=sensitive_attr)
        report['False_Positive_Rate_Parity'] = false_positive_rate_parity(y_true=y_true, y_pred=y_pred, sensitive_attr=sensitive_attr)
        report['Predictive_Parity'] = predictive_parity(y_true=y_true, y_pred=y_pred, sensitive_attr=sensitive_attr)
        report['Treatment_Equality'] = treatment_equality(y_true=y_true, y_pred=y_pred, sensitive_attr=sensitive_attr)
        
        # Log the metrics for debugging
        logging.info("Fairness metrics calculated successfully:")
        for k, v in report.items():
            logging.info(f"{k}: {v}")
            
        return report  # Return the complete dictionary
        
    except Exception as e:
        logging.error(f"Error in generate_report_fairness: {str(e)}")
        raise CustomException(e, sys)


if __name__ == "__main__": 
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    try:
        df = pd.read_csv("dfs/Adult.csv")    
        target_col = "income"
        sensitive_col = "sex"
        
        # Map target to binary
        df[target_col] = df[target_col].map({df[target_col].unique()[0]: 0, df[target_col].unique()[1]: 1})

        # Map sensitive attribute to binary
        df[sensitive_col] = df[sensitive_col].map({df[sensitive_col].unique()[0]: 0, df[sensitive_col].unique()[1]: 1})

        # Prepare features
        X = pd.get_dummies(df.drop(columns=[target_col, sensitive_col]), drop_first=True)
        y = df[target_col].values
        s = df[sensitive_col].values

        # Train/test split
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, s, test_size=0.3, random_state=42
        )

        # Train model
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Run fairness report
        report = generate_report_fairness(y_test, y_pred, s_test)
        
        print("\n📊 Fairness Report:")
        for k, v in report.items():
            print(f"{k}: {v}")
            
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        print(f"Error: {e}")
        
        
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
