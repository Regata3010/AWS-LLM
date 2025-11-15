import sys
from core.src.logger import logging
from core.src.exception import CustomException
import pandas as pd
from core.src.ingestion import download_file_from_s3
import os

def profile_dataset(file_path:str):
    try:
        df = pd.read_csv(file_path)
        logging.info("Dataset loaded successfully and the shape is {}".format(df.shape))
        
        profile = {}
        
        profile['shape'] = df.shape
        profile['null_values'] = df.isnull().sum().to_dict()
        profile['dtypes'] = df.dtypes.astype(str).to_dict()
        profile['cardinality'] = df.nunique().to_dict()
 
        profile['categorical_distributions'] = {}
        for col in df.columns:
            if df[col].nunique() <= 10:
                profile['categorical_distributions'][col] = (
                    df[col].value_counts(normalize=True, dropna=False).round(4).to_dict()
                )
                
        potential_labels = [col for col in df.columns if df[col].nunique() <= 2 and df[col].dtype in ['int64','float64']]
        if potential_labels:
            label_col = potential_labels[0]
            profile['class_imbalance'] = df[label_col].value_counts(normalize=True).round(4).to_dict()
        
        logging.info(f"Profile : {profile}")  #imp      
        return profile
            
    
    except Exception as e:
        logging.error(f"Error profiling dataset: {e}")
        raise CustomException(e,sys)
    

# if __name__=="__main__":
#     dir_path = "/Users/AWS-LLM/dfs"
#     os.makedirs(dir_path, exist_ok=True)
#     local_file_path = os.path.join(dir_path, "Adult.csv")
#     file_path = download_file_from_s3(bucket="qwezxcbucket",s3_key="Adult.csv",file_path=local_file_path)
#     profile_dataset(file_path=file_path)