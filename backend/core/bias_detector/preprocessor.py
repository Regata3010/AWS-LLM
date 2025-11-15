import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from core.src.logger import logging
from core.src.exception import CustomException
import sys
from typing import Union, List
from core.src.ingestion import download_file_from_s3
from core.llms.column_selector import final_columns_from_csv

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drops NA values"""
    try:
        logging.info("Cleaning data...")
        return df.dropna()
    except Exception as e:
        raise CustomException(e,sys)
    
def split_data(X, y, s, test_size=0.2, random_state=42):
    """
    Splits X, y, and sensitive attributes s into train and test sets.

    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix
        y (pd.Series or np.ndarray): Target variable
        s (np.ndarray): Sensitive attribute(s)
        test_size (float): Test set proportion
        random_state (int): Random seed

    Returns:
        Tuple: X_train, X_test, y_train, y_test, s_train, s_test
    """
    try:
        # Split features and target first
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, s, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
        )

        return X_train, X_test, y_train, y_test, s_train, s_test

    except Exception as e:
        raise CustomException(e,sys)


def encode_data(df: pd.DataFrame, target_col: str, sensitive_col: Union[str, List[str]]):
    """
    Encodes the target column and one-hot encodes the remaining features, 
    excluding target and sensitive columns.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        target_col (str): Name of the target column.
        sensitive_col (Union[str, List[str]]): Column(s) considered sensitive.

    Returns:
        X (pd.DataFrame): Features after encoding.
        y (np.array): Encoded target.
        s (np.array): Sensitive attributes.
    """
    try:
       
        sensitive_cols = [sensitive_col] if isinstance(sensitive_col, str) else sensitive_col

        
        missing_cols = [col for col in [target_col] + sensitive_cols if col not in df.columns]
        if missing_cols:
            raise CustomException(f"❌ Column(s) not found in DataFrame: {missing_cols}",sys)

        # ✅ Encode target column if it's categorical
        y = df[target_col]
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        # ✅ Extract sensitive attributes
        s = df[sensitive_cols].copy()
        s = s.values
        if s.ndim == 2 and s.shape[1] == 1:
            s = s.flatten()

        # ✅ Drop target + sensitive columns from features
        X = df.drop(columns=[target_col] + sensitive_cols)

        # ✅ One-hot encode remaining columns
        if X.empty:
            raise CustomException("❌ No features left after dropping target and sensitive columns.",sys)

        X = pd.get_dummies(X, drop_first=True)

        return X, y, s

    except Exception as e:
        raise CustomException(e,sys)

def preprocess_data(df: pd.DataFrame, target_col: str, sensitive_col: Union[str,List[str]], test_size=0.2, random_state=42):
    """
    Main wrapper that preprocesses the dataset
    """
    try:
        df = clean_data(df)
        X, y, s = encode_data(df, target_col, sensitive_col)
        X_train, X_test, y_train, y_test, s_train, s_test = split_data(X, y, s, test_size, random_state)
        
        logging.info("Data preprocessing complete.")
        return X_train, X_test, y_train, y_test, s_train, s_test
    except Exception as e:
        raise CustomException(e,sys)


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("dfs/Adult.csv")
    # print(df.columns.tolist())
    
    target_col = "income"
    sensitive_col = ["sex", "race"]
    try:
        X_train, X_test, y_train, y_test, s_train, s_test = preprocess_data(df, target_col, sensitive_col)
        print("✅ Preprocessing succeeded")
        print("X_train shape:", X_train.shape)
        print("y_train sample:", y_train[:5])
        print("s_train sample:", s_train[:5])
    except Exception as e:
        print("❌ Preprocessing failed:", e)