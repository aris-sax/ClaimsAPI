from enum import Enum


class TaskStatus(Enum):
    PENDING = "Pending"
    PROCESSING = "Processing"
    COMPLETE = "Complete"
    FAILED = "Failed"


class FilePurposeLLM(Enum):
    ASSISTANTS = "assistants"
    BATCH = "batch"
    FINE_TUNE = "fine-tune"


class FileType(Enum):
    PDF = "pdf"
    IMAGE = "image"
    DOCX = "docx"
    TXT = "txt"

