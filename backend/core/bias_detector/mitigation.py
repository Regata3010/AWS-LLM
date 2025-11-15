from core.src.logger import logging
from core.src.exception import CustomException
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import RejectOptionClassification
from aif360.metrics import BinaryLabelDatasetMetric
import pandas as pd
import sys
from sklearn.linear_model import LinearRegression
from core.llms.column_selector import final_columns_from_csv



def convert_to_aif360(df, label_col:str, sensitive_col:str):
    try:
        dataset = StandardDataset(
            df,
            label_name=label_col,
            favorable_classes=[1],
            protected_attribute_names=[sensitive_col],
            privileged_classes=[[1]],
        )
        return dataset
    except Exception as e:
        logging.info("there has been an Problem")
        raise CustomException(e,sys)
    

def convert_to_dataframe(aif_dataset) -> pd.DataFrame:
    try:
        df, _ = aif_dataset.convert_to_dataframe()
        return df 
    except Exception as e:
        logging.info("There has been a Problem")
        raise CustomException(e,sys)


def apply_reweighing(df,label_col:str,sensitive_col:str):
    try:
        dataset = convert_to_aif360(df,label_col=label_col,sensitive_col=sensitive_col)
        RW = Reweighing(unprivileged_groups=[{sensitive_col : 0}],
                        privileged_groups=[{sensitive_col : 1}])
        
        mitigated = RW.fit_transform(dataset)
        return mitigated
    except Exception as e:
        logging.info("There has been a Problem")
        raise CustomException(e,sys)
        
def apply_reject_option(test_dataset, sensitive_col):
    try:
        roc = RejectOptionClassification(
            unprivileged_groups=[{sensitive_col:0}],
            privileged_groups=[{sensitive_col:1}],
            low_class_thresh=0.01, high_class_thresh=0.99,
            num_class_thresh=100, num_ROC_margin=50,
            metric_name="Statistical Parity Difference",
            metric_ub=0.05,metric_lb=-0.05
        )
        
        roc = roc.fit(test_dataset, test_dataset)
        return roc.predict(test_dataset)
    except Exception as e:
        logging.info("There has been an a Problem")
        raise CustomException(e,sys)