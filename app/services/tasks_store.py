import asyncio

from fastapi import HTTPException

from app.config import settings
from app.pydantic_schemas.claims_extraction_task import ClaimsExtractionTask
from app.utils.enums import TaskStatus


class ClaimsExtractionStore:
    def __init__(self):
        pass

    @staticmethod
    def get_task_data_from_tracker(task_uuid: str):
        task_data = settings.CLAIMS_CHECKER_RESPONSES_TRACKER.get(task_uuid)
        if not task_data:
            raise HTTPException(status_code=404, detail="Annotations Extraction Task Not Found")
        return task_data

    @staticmethod
    def store_task_in_tracker(task: ClaimsExtractionTask) -> None:
        settings.CLAIMS_CHECKER_RESPONSES_TRACKER[task.task_uuid] = task
        print("Total number of tasks in AnnotationsExtractionTask tracker: ",
              len(settings.CLAIMS_CHECKER_RESPONSES_TRACKER))

    @staticmethod
    def remove_task_from_tracker(task_data: ClaimsExtractionTask) -> None:
        if task_data.task_status == TaskStatus.COMPLETE:
            settings.CLAIMS_CHECKER_RESPONSES_TRACKER.pop(task_data.task_uuid, None)

    @staticmethod
    async def remove_task_after_delay(task_data: ClaimsExtractionTask, delay: int):
        await asyncio.sleep(delay)
        ClaimsExtractionStore.remove_task_from_tracker(task_data)