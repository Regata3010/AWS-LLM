from core.src.exception import CustomException 
from core.src.logger import logging
import pandas as pd
import numpy as np
import sys


def type_of_target(y_train):
    try:
        y = pd.Series(y_train)  

        unique_values = y.unique()
        n_unique = len(unique_values)

        if y.dtype.kind in 'ifu':  # integer, float, unsigned
            if n_unique <= 10 and set(unique_values).issubset({0, 1}):
                return "binary"
            elif n_unique <= 20:
                return "multiclass"
            else:
                return "continuous"
        elif y.dtype == 'object' or y.dtype.name == 'category':
            if n_unique == 2:
                return "binary"
            else:
                return "multiclass"
        else:
            return "unknown"

    except Exception as e:
        logging.info("There has been a Problem")
        raise CustomException(e,sys)
    
    
# if __name__=="__main__":
    # y1 = [0, 1, 1, 0, 1]
    # y2 = [1, 2, 3, 2, 1, 3]
    # y3 = [2.5, 3.7, 4.1, 5.0]
    # y4 = ['yes', 'no', 'yes', 'no']

    # print(type_of_target(y1))  # ➤ binary
    # print(type_of_target(y2))  # ➤ multiclass
    # print(type_of_target(y3))  # ➤ continuous
    # print(type_of_target(y4))  # ➤ binary
