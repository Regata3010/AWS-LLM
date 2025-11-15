import sys
import types
from typing import Optional
# from core.src.logger import logging

def error_message_detail(error, error_detail: types.ModuleType):
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = "Error has Occured in Python Script [{0}] line number [{1}] error message [{2}]".format(
            file_name, exc_tb.tb_lineno, str(error))
    else:
        error_message = f"Error occurred: {str(error)} (traceback not available)"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: types.ModuleType):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
    
    def __str__(self):
        return self.error_message


if __name__=="__main__":
    try:
        a = 1/0
        
    except Exception as e:
        # logging.info("Divide By Zero.")
        raise CustomException(e,sys)