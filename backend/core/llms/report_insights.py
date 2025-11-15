from src.exception import CustomException
from src.logger import logging
from dotenv import load_dotenv
import sys, os
import openai


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


try:
    pass

except Exception as e:
    logging.info("There has been an issue.")
    raise CustomException(e,sys)