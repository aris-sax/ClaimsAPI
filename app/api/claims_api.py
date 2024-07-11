from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, UploadFile, HTTPException

from app.pydantic_schemas.apis_response_utils import TaskUuidResponse
from app.pydantic_schemas.claims_extraction_task import ClaimsExtractionTask, ClaimResult, DocumentJsonFormat
from app.services.claims_extraction_service import ClaimsExtractionService
from app.services.tasks_store import ClaimsExtractionStore
from app.utils.utils import validate_file

router = APIRouter()
claims_extraction_store = ClaimsExtractionStore()


@router.post("/request", response_model=TaskUuidResponse)
async def request_files_review(clinical_file: Annotated[UploadFile, File(description="Clinical file")],
                               background_tasks: BackgroundTasks):
    allowed_extensions = {".pdf"}
    try:
        # Validate clinical files
        validate_file(clinical_file, allowed_extensions)
        annotation_extraction_task = ClaimsExtractionService.create_and_store_task(clinical_file)
        print(f"Created task: {annotation_extraction_task.task_uuid}")
        # Add background task to run the review workflow
        background_tasks.add_task(ClaimsExtractionService(annotation_extraction_task).run)
        return TaskUuidResponse(task_uuid=annotation_extraction_task.task_uuid)
    except HTTPException as e:
        # Raise HTTPException for client errors
        raise e
    except Exception as e:
        # Log the error for debugging purposes
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please review the logs.")


@router.post("/results", response_model=ClaimsExtractionTask)
async def get_review_results(task_uuid: str, background_tasks: BackgroundTasks):
    try:
        results = claims_extraction_store.get_task_data_from_tracker(task_uuid)
        print(f"Retrieved task: {results.task_uuid}")
        task_dict = results.model_dump(exclude={"task_document"})
        # Schedule the task removal after 5 minutes (300 seconds)
        background_tasks.add_task(ClaimsExtractionStore.remove_task_after_delay, results, 300)
        return task_dict
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please review the logs.")
    
    
@router.post("/view_formatted_document", response_model=DocumentJsonFormat)
async def get_pdf_formatted_with_metadata(task_uuid: str):
    try:
        results = claims_extraction_store.get_task_data_from_tracker(task_uuid)
        print(f"Retrieved task: {results.task_uuid}")
        return results.task_document.text_file_with_metadata
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please review the logs.")
