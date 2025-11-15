import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # S3 Configuration
    S3_BUCKET = os.getenv('S3_BUCKET', 'qwezxcbucket')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

settings = Settings()

# In upload.py:
# from config.settings import settings
# BUCKET_NAME = settings.S3_BUCKET