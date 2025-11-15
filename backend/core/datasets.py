# from fastapi import APIRouter, UploadFile, File, HTTPException
# from backend.services.dataset_service import DatasetService

# router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

# # Initialize service
# dataset_service = DatasetService()

# @router.post("/upload")
# async def upload_dataset(file: UploadFile = File(...)):
#     """
#     Upload CSV dataset with comprehensive quality assessment and LLM column detection.
    
#     Returns:
#     - Dataset metadata
#     - Data quality score (0-100)
#     - LLM column detection results
#     - Analysis readiness status
#     """
#     try:
#         result = await dataset_service.upload_and_process(file)
#         return result
#     except ValueError as e:
#         # Client errors (bad file, etc.)
#         raise HTTPException(status_code=400, detail=str(e))
#     except RuntimeError as e:
#         # Server errors (S3 issues, etc.)
#         raise HTTPException(status_code=500, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# @router.get("/{dataset_id}")
# async def get_dataset_info(dataset_id: str):
#     """
#     Get comprehensive dataset information including:
#     - Metadata and file info
#     - Data quality assessment results
#     - Column detection results
#     - Data preview
#     """
#     try:
#         result = await dataset_service.get_dataset_info(dataset_id)
#         return result
#     except FileNotFoundError:
#         raise HTTPException(status_code=404, detail="Dataset not found")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to retrieve dataset info: {str(e)}")

# @router.get("/")
# async def list_datasets():
#     """
#     List all uploaded datasets with:
#     - Basic metadata
#     - Quality scores
#     - Analysis readiness status
#     - Column detection success
#     """
#     try:
#         result = await dataset_service.list_all_datasets()
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")

# @router.delete("/{dataset_id}")
# async def delete_dataset(dataset_id: str):
#     """Delete dataset and all associated metadata from S3."""
#     try:
#         result = await dataset_service.delete_dataset(dataset_id)
#         return result
#     except FileNotFoundError:
#         raise HTTPException(status_code=404, detail="Dataset not found")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")

# @router.get("/{dataset_id}/quality")
# async def get_dataset_quality_report(dataset_id: str):
#     """
#     Get detailed data quality assessment for a dataset.
#     Useful for understanding why a dataset might not be ready for analysis.
#     """
#     try:
#         metadata = await dataset_service.get_dataset_info(dataset_id)
        
#         quality_report = metadata.get('data_quality', {})
#         if not quality_report:
#             raise HTTPException(status_code=404, detail="Quality report not found")
        
#         return {
#             'dataset_id': dataset_id,
#             'filename': metadata.get('filename'),
#             'quality_assessment': quality_report,
#             'ready_for_analysis': metadata.get('ready_for_bias_analysis', False)
#         }
#     except FileNotFoundError:
#         raise HTTPException(status_code=404, detail="Dataset not found")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to get quality report: {str(e)}")

# @router.get("/{dataset_id}/columns")
# async def get_dataset_columns_info(dataset_id: str):
#     """
#     Get detailed column detection results for a dataset.
#     Shows LLM reasoning and detected target/sensitive columns.
#     """
#     try:
#         metadata = await dataset_service.get_dataset_info(dataset_id)
        
#         column_info = {
#             'dataset_id': dataset_id,
#             'filename': metadata.get('filename'),
#             'all_columns': metadata.get('columns', []),
#             'column_types': metadata.get('dtypes', {}),
#             'llm_detection': metadata.get('column_detection', {}),
#             'ready_for_analysis': metadata.get('ready_for_bias_analysis', False)
#         }
        
#         return column_info
#     except FileNotFoundError:
#         raise HTTPException(status_code=404, detail="Dataset not found")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to get column info: {str(e)}")