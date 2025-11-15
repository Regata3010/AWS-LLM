import pandas as pd
import io
import json
import uuid
import boto3
import os, sys
from datetime import datetime
from dotenv import load_dotenv
from fastapi import UploadFile

from core.llms.column_selector import api_column_detection
from core.src.ingestion import upload_file_to_s3, download_file_from_s3
from core.src.exception import CustomException

load_dotenv()

class DatasetService:
    """
    Business logic for dataset management.
    Focuses on S3 operations, data quality, and orchestration.
    Column detection handled by column_selector.py
    """
    
    def __init__(self):
        self.s3_bucket = "qwezxcbucket"
        self.metadata_prefix = "metadata/"
        self.data_prefix = "datasets/"
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
    
    async def upload_and_process(self, file: UploadFile):
        """
        Upload CSV to S3 with proper workflow:
        1. Validate file
        2. Assess data quality 
        3. Column detection (only if quality is good)
        4. Store with comprehensive metadata
        """
        try:
            # Step 1: File validation
            if not file.filename:
                raise ValueError("No filename provided")
                
            if not file.filename.endswith('.csv'):
                raise ValueError("Only CSV files are supported")
            
            # Read and validate CSV
            file_content = await file.read()
            if len(file_content) == 0:
                raise ValueError("Empty file uploaded")
                
            file_buffer = io.BytesIO(file_content)
            
            try:
                df = pd.read_csv(file_buffer)
                if df.empty:
                    raise ValueError("CSV file is empty")
                if len(df.columns) < 2:
                    raise ValueError("CSV must have at least 2 columns for bias analysis")
            except pd.errors.EmptyDataError:
                raise ValueError("CSV file is empty or corrupted")
            except Exception as e:
                raise ValueError(f"Invalid CSV file: {str(e)}")
            
            # Step 2: COMPREHENSIVE Data Quality Assessment (BEFORE column detection)
            data_quality = self._assess_data_quality(df)
            
            # Step 3: Column detection (ONLY if data quality is acceptable)
            column_detection = {}
            if data_quality.get('is_suitable', False):
                try:
                    # Use your existing column_selector.py - no duplication!
                    column_detection = api_column_detection(df)
                except Exception as e:
                    column_detection = {
                        'success': False,
                        'error': str(e),
                        'target_column': None,
                        'sensitive_columns': [],
                        'method': 'failed'
                    }
            else:
                column_detection = {
                    'success': False,
                    'error': 'Data quality too poor for column detection',
                    'target_column': None,
                    'sensitive_columns': [],
                    'method': 'skipped_due_to_quality'
                }
            
            # Generate S3 keys
            dataset_id = str(uuid.uuid4())
            data_s3_key = f"{self.data_prefix}{dataset_id}_{file.filename}"
            metadata_s3_key = f"{self.metadata_prefix}{dataset_id}_metadata.json"
            
            # Upload CSV to S3
            file_buffer.seek(0)
            try:
                upload_file_to_s3(file_obj=file_buffer, bucket=self.s3_bucket, s3_key=data_s3_key)
            except Exception as e:
                raise RuntimeError(f"S3 upload failed: {str(e)}")
            
            # comprehensive metadata
            metadata = {
                'dataset_id': dataset_id,
                'filename': file.filename,
                'data_s3_key': data_s3_key,
                'metadata_s3_key': metadata_s3_key,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'data_quality': data_quality,  # FIRST
                'column_detection': column_detection,  # SECOND (based on quality)
                'upload_timestamp': datetime.now().isoformat(),
                'file_size_bytes': len(file_content),
                'ready_for_bias_analysis': (
                    data_quality.get('is_suitable', False) and 
                    column_detection.get('success', False)
                )
            }
            
            # Store metadata in S3
            try:
                metadata_json = json.dumps(metadata, indent=2)
                metadata_buffer = io.BytesIO(metadata_json.encode('utf-8'))
                upload_file_to_s3(file_obj=metadata_buffer, bucket=self.s3_bucket, s3_key=metadata_s3_key)
            except Exception as e:
                raise RuntimeError(f"Metadata upload failed: {str(e)}")
            
            # Return comprehensive response
            return {
                'dataset_id': dataset_id,
                'filename': file.filename,
                'rows': df.shape[0],
                'columns': df.shape[1],
                'column_names': df.columns.tolist(),
                'data_quality': data_quality,
                'llm_detection': column_detection,
                'status': 'uploaded_successfully',
                's3_location': data_s3_key,
                'ready_for_analysis': metadata['ready_for_bias_analysis']
            }
            
        except (ValueError, RuntimeError) as e:
            raise e
        except Exception as e:
            raise CustomException(e, sys)
    
    def _assess_data_quality(self, df: pd.DataFrame) -> dict:
        """
        COMPREHENSIVE data quality assessment - done FIRST before any other processing.
        Determines if dataset is suitable for ML bias analysis.
        """
        try:
            quality_report = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_data_analysis': {},
                'data_type_analysis': {},
                'statistical_summary': {},
                'quality_issues': [],
                'quality_score': 0,  # 0-100 score
                'is_suitable': True,
                'recommendations': []
            }
            
            # 1. MISSING DATA ANALYSIS
            total_missing = 0
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_percentage = (missing_count / len(df)) * 100
                total_missing += missing_count
                
                quality_report['missing_data_analysis'][col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_percentage, 2)
                }
                
                # Flag missing data issues
                if missing_percentage > 30:
                    quality_report['quality_issues'].append(
                        f"Column '{col}' has {missing_percentage:.1f}% missing values - consider imputation"
                    )
                    if missing_percentage > 70:
                        quality_report['quality_issues'].append(
                            f"Column '{col}' has {missing_percentage:.1f}% missing - may need to drop"
                        )
            
            # 2. DATA TYPE ANALYSIS
            numeric_cols = 0
            categorical_cols = 0
            high_cardinality_cols = 0
            
            for col in df.columns:
                unique_count = df[col].nunique()
                unique_percentage = (unique_count / len(df)) * 100
                dtype = str(df[col].dtype)
                
                analysis = {
                    'dtype': dtype,
                    'unique_values': int(unique_count),
                    'unique_percentage': round(unique_percentage, 2),
                    'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
                    'is_categorical': unique_count <= 20 or dtype == 'object',
                    'is_binary': unique_count == 2,
                    'is_high_cardinality': unique_percentage > 95  # Likely ID columns
                }
                
                quality_report['data_type_analysis'][col] = analysis
                
                # Count column types
                if analysis['is_numeric']:
                    numeric_cols += 1
                if analysis['is_categorical']:
                    categorical_cols += 1
                if analysis['is_high_cardinality']:
                    high_cardinality_cols += 1
                    quality_report['quality_issues'].append(
                        f"Column '{col}' has {unique_percentage:.1f}% unique values - likely ID column"
                    )
            
            # 3. STATISTICAL SUMMARY
            if numeric_cols > 0:
                try:
                    numeric_df = df.select_dtypes(include=['number'])
                    # Calculate statistics safely with proper type handling
                    try:
                        mean_series = numeric_df.mean()
                        std_series = numeric_df.std()
                        
                        # Convert Series to dict, handling edge cases
                        if isinstance(mean_series, pd.Series):
                            mean_values = mean_series.to_dict()
                        else:
                            mean_values = {}
                            
                        if isinstance(std_series, pd.Series):
                            std_values = std_series.to_dict()
                        else:
                            std_values = {}
                            
                        zero_var_count = len(numeric_df.columns[numeric_df.std() == 0])
                    except Exception:
                        mean_values = {}
                        std_values = {}
                        zero_var_count = 0
                    
                    quality_report['statistical_summary'] = {
                        'numeric_columns': numeric_cols,
                        'mean_values': mean_values,
                        'std_values': std_values,
                        'zero_variance_columns': zero_var_count
                    }
                    
                    # Flag zero variance columns
                    try:
                        zero_var_cols = numeric_df.columns[numeric_df.std() == 0].tolist()
                    except Exception:
                        zero_var_cols = []
                    if zero_var_cols:
                        quality_report['quality_issues'].append(
                            f"Columns with zero variance: {zero_var_cols} - consider removing"
                        )
                except:
                    quality_report['statistical_summary'] = {'error': 'Could not compute statistics'}
            
            # 4. DATASET SIZE VALIDATION
            if len(df) < 100:
                quality_report['quality_issues'].append(
                    f"Dataset has only {len(df)} rows - recommend at least 100 for ML"
                )
                if len(df) < 50:
                    quality_report['is_suitable'] = False
            
            if len(df.columns) < 3:
                quality_report['quality_issues'].append(
                    "Dataset needs at least 3 columns (target + features + sensitive attributes)"
                )
                quality_report['is_suitable'] = False
            
            # 5. ML SUITABILITY CHECKS
            potential_targets = 0
            potential_sensitive = 0
            
            for col in df.columns:
                col_lower = col.lower()
                unique_count = df[col].nunique()
                
                # Check for potential target columns (2-10 unique values)
                if 2 <= unique_count <= 10:
                    potential_targets += 1
                
                # Check for potential sensitive attributes
                sensitive_keywords = ['age', 'gender', 'sex', 'race', 'ethnicity', 'education']
                if any(keyword in col_lower for keyword in sensitive_keywords):
                    potential_sensitive += 1
            
            if potential_targets == 0:
                quality_report['quality_issues'].append(
                    "No columns suitable as target variable (need 2-10 unique values)"
                )
                quality_report['is_suitable'] = False
            
            if potential_sensitive == 0:
                quality_report['quality_issues'].append(
                    "No obvious sensitive attributes detected (age, gender, race, etc.)"
                )
            
            # 6. CALCULATE QUALITY SCORE (0-100)
            score = 100
            
            # Deduct for missing data
            overall_missing_rate = (total_missing / (len(df) * len(df.columns))) * 100
            score -= min(overall_missing_rate, 30)
            
            # Deduct for high cardinality columns
            score -= (high_cardinality_cols * 10)
            
            # Deduct for size issues
            if len(df) < 100:
                score -= 20
            if len(df.columns) < 5:
                score -= 15
            
            # Deduct for suitability issues
            if potential_targets == 0:
                score -= 25
            if potential_sensitive == 0:
                score -= 10
            
            quality_report['quality_score'] = max(int(score), 0)
            
            # 7. GENERATE RECOMMENDATIONS
            if quality_report['quality_score'] >= 80:
                quality_report['recommendations'].append("Dataset quality is excellent for bias analysis")
            elif quality_report['quality_score'] >= 60:
                quality_report['recommendations'].append("Dataset quality is good with minor issues")
                quality_report['recommendations'].extend([
                    "Consider addressing missing data issues",
                    "Remove high-cardinality ID columns if present"
                ])
            else:
                quality_report['recommendations'].extend([
                    "Dataset quality needs improvement before bias analysis",
                    "Address missing data issues (imputation or removal)",
                    "Ensure adequate sample size (recommended: 500+ rows)",
                    "Verify presence of suitable target and sensitive attributes"
                ])
                if quality_report['quality_score'] < 40:
                    quality_report['is_suitable'] = False
            
            return quality_report
            
        except Exception as e:
            return {
                'error': str(e),
                'is_suitable': False,
                'quality_issues': ['Data quality assessment failed'],
                'quality_score': 0,
                'recommendations': ['Manual data review required']
            }
    
    async def load_dataset_for_analysis(self, dataset_id: str) -> tuple[pd.DataFrame, dict]:
        """
        Load dataset from S3 for bias analysis.
        Used by bias detection service.
        """
        try:
            # Get metadata
            metadata_s3_key = f"{self.metadata_prefix}{dataset_id}_metadata.json"
            metadata_buffer = download_file_from_s3(bucket=self.s3_bucket, s3_key=metadata_s3_key)
            metadata = json.loads(metadata_buffer.getvalue().decode('utf-8'))
            
            # Check if ready for analysis
            if not metadata.get('ready_for_bias_analysis', False):
                quality_score = metadata.get('data_quality', {}).get('quality_score', 0)
                column_success = metadata.get('column_detection', {}).get('success', False)
                
                error_details = []
                if quality_score < 60:
                    error_details.append(f"Data quality score too low: {quality_score}/100")
                if not column_success:
                    error_details.append("Column detection failed")
                
                raise ValueError(f"Dataset not ready for analysis: {'; '.join(error_details)}")
            
            # Load data
            data_buffer = download_file_from_s3(bucket=self.s3_bucket, s3_key=metadata['data_s3_key'])
            df = pd.read_csv(data_buffer)
            
            return df, metadata
            
        except Exception as e:
            if "NoSuchKey" in str(e):
                raise FileNotFoundError("Dataset not found")
            raise CustomException(e, sys)
    
    async def get_dataset_info(self, dataset_id: str):
        """Get comprehensive dataset information."""
        try:
            # Load metadata
            metadata_s3_key = f"{self.metadata_prefix}{dataset_id}_metadata.json"
            metadata_buffer = download_file_from_s3(bucket=self.s3_bucket, s3_key=metadata_s3_key)
            metadata = json.loads(metadata_buffer.getvalue().decode('utf-8'))
            
            # Load data for preview
            try:
                data_buffer = download_file_from_s3(bucket=self.s3_bucket, s3_key=metadata['data_s3_key'])
                df = pd.read_csv(data_buffer)
                
                # Enhanced data preview
                metadata['data_preview'] = {
                    'head': df.head(3).to_dict('records'),
                    'missing_values_summary': df.isnull().sum().to_dict(),
                    'sample_distributions': {
                        col: df[col].value_counts().head(3).to_dict() 
                        for col in df.columns 
                        if df[col].dtype == 'object' and df[col].nunique() <= 10
                    }
                }
            except Exception as e:
                metadata['data_preview'] = {'error': str(e)}
            
            return metadata
            
        except Exception as e:
            if "NoSuchKey" in str(e):
                raise FileNotFoundError("Dataset not found")
            raise CustomException(e, sys)
    
    async def list_all_datasets(self):
        """List all datasets with quality and analysis readiness."""
        try:
            # List metadata files
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=self.metadata_prefix
            )
            
            datasets_summary = []
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    try:
                        # Download metadata
                        metadata_obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=obj['Key'])
                        metadata = json.loads(metadata_obj['Body'].read().decode('utf-8'))
                        
                        summary = {
                            'dataset_id': metadata['dataset_id'],
                            'filename': metadata['filename'],
                            'shape': metadata['shape'],
                            'upload_timestamp': metadata['upload_timestamp'],
                            'data_quality_score': metadata.get('data_quality', {}).get('quality_score', 0),
                            'llm_detection_success': metadata.get('column_detection', {}).get('success', False),
                            'target_column': metadata.get('column_detection', {}).get('target_column'),
                            'sensitive_columns': metadata.get('column_detection', {}).get('sensitive_columns', []),
                            'file_size_mb': round(metadata.get('file_size_bytes', 0) / (1024*1024), 2),
                            'ready_for_analysis': metadata.get('ready_for_bias_analysis', False)
                        }
                        datasets_summary.append(summary)
                        
                    except Exception:
                        continue  # Skip corrupted files
            
            return {
                'total_datasets': len(datasets_summary),
                'datasets': sorted(datasets_summary, key=lambda x: x['upload_timestamp'], reverse=True),
                'ready_for_analysis': len([d for d in datasets_summary if d['ready_for_analysis']]),
                'average_quality_score': round(
                    sum(d['data_quality_score'] for d in datasets_summary) / len(datasets_summary), 1
                ) if datasets_summary else 0
            }
            
        except Exception as e:
            raise CustomException(e, sys)
    
    async def delete_dataset(self, dataset_id: str):
        """Delete dataset and metadata from S3."""
        try:
            # Get metadata first
            metadata_s3_key = f"{self.metadata_prefix}{dataset_id}_metadata.json"
            
            try:
                metadata_obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=metadata_s3_key)
                metadata = json.loads(metadata_obj['Body'].read().decode('utf-8'))
                data_s3_key = metadata['data_s3_key']
                filename = metadata['filename']
            except:
                raise FileNotFoundError("Dataset not found")
            
            # Delete from S3
            self.s3_client.delete_object(Bucket=self.s3_bucket, Key=data_s3_key)
            self.s3_client.delete_object(Bucket=self.s3_bucket, Key=metadata_s3_key)
            
            return {
                'message': f"Dataset '{filename}' deleted successfully",
                'dataset_id': dataset_id,
                'deleted_files': [data_s3_key, metadata_s3_key]
            }
            
        except FileNotFoundError:
            raise
        except Exception as e:
            raise CustomException(e, sys)