from pydantic import BaseModel


class BasicAPIResponse(BaseModel):
    response: str


class TaskUuidResponse(BaseModel):
    task_uuid: str
