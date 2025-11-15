# backend/api/routes/upload.py
from fastapi import APIRouter, File, UploadFile, HTTPException
import boto3
import uuid
import io
import pandas as pd
import sys
from configurations.settings import settings
from api.models.responses import DatasetUploadResponse
from api.models.requests import DatasetUploadRequests
from core.src.logger import logging
from core.src.exception import CustomException
import mlflow
from deprecated import deprecated

router = APIRouter()


s3 = boto3.client(
    's3',
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
)

#added a extra config file for better creds handling
BUCKET_NAME = settings.S3_BUCKET

@router.post("/upload", include_in_schema=False, response_model=DatasetUploadResponse)
@deprecated(reason="Use BiasGuard 2.0 dataset upload workflow instead")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload CSV for bias analysis"""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files allowed")
        
        # Generate unique file ID
        file_id = f"{uuid.uuid4().hex}"
        s3_key = f"datasets/{file_id}/{file.filename}"
        
        # Read and validate CSV
        content = await file.read()
        file_obj = io.BytesIO(content)
        df = pd.read_csv(file_obj)
        
        # Upload to S3
        file_obj.seek(0)
        s3.upload_fileobj(file_obj, BUCKET_NAME, s3_key)
        
        logging.info(f"Uploaded to s3://{BUCKET_NAME}/{s3_key}")
        
        return DatasetUploadResponse(
            file_id=file_id,
            filename=file.filename,
            rows=len(df),
            columns=len(df.columns),
            # column_names = df.columns.to_list(),
            s3_path=f"s3://{BUCKET_NAME}/{s3_key}",
            upload_timestamp=pd.Timestamp.now().isoformat(),
            file_size_mb=round(len(content) / (1024 * 1024), 2)
        )
        
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=410, detail=f"Invalid CSV: {str(e)}")
    except Exception as e:
        logging.error(f"Upload error: {e}")
        raise HTTPException(status_code=410, detail="=Upload endpoint deprecated. Use BiasGuard 2.0 dataset upload workflow.")

# Helper function for internal use by other routers
async def get_file_from_s3(file_id: str) -> pd.DataFrame:
    """Internal helper - not an endpoint"""
    try:
        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=f"datasets/{file_id}/"
        )
        
        if 'Contents' not in response:
            raise ValueError(f"File {file_id} not found")
        
        s3_key = response['Contents'][0]['Key']
        
        buffer = io.BytesIO()
        s3.download_fileobj(BUCKET_NAME, s3_key, buffer)
        buffer.seek(0)
        
        return pd.read_csv(buffer)
        
    except Exception as e:
        logging.error(f"Error getting file from S3: {e}")
        raise CustomException(e, sys)