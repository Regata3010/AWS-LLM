import boto3
from dotenv import load_dotenv
import os
from core.src.logger import logging
from core.src.exception import CustomException
import sys
import io
import pandas as pd

load_dotenv()

s3 = boto3.client('s3',aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                  aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))


def upload_file_to_s3(file_obj,bucket,s3_key):
    try:
        file_obj.seek(0)
        s3.upload_fileobj(file_obj,bucket,s3_key)
        logging.info(f"Uploaded {file_obj} to s3://{bucket}/{s3_key}")
    except Exception as e:
        logging.error(f"Error uploading file to s3: {e}")
        raise CustomException(e,sys)


def download_file_from_s3(bucket, s3_key):
    try:
        buffer = io.BytesIO()
        s3.download_fileobj(bucket, s3_key, buffer)
        buffer.seek(0)

        if buffer.getbuffer().nbytes == 0:
            raise ValueError("Downloaded file is empty")

        logging.info(f"Downloaded {s3_key} from s3://{bucket}")
        return buffer

    except Exception as e:
        logging.error(f"Download failed: {e}")
        raise CustomException(e, sys)


    
def list_files_in_bucket(bucket, prefix=""):
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return [item['Key'] for item in response.get('Contents', [])]
    except Exception as e:
        # logging.error(f"Error listing files in S3: {e}")
        raise CustomException(e, sys)

# if __name__=="__main__":
#     #we can call the upload or download function as per requirements
# #     pass

# if __name__ == "__main__":
#     upload_file_to_s3(file_path="/Users/mll/adult.csv",bucket="qwezxcbucket",s3_key="Adult.csv")