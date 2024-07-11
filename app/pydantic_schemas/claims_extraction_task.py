from typing import List, Any, Dict, Optional, Union, Literal

from pydantic import BaseModel

from app.utils.enums import TaskStatus


class LineRange(BaseModel):
    start: int
    end: int
    

class PageContentItem(BaseModel):
    text: str
    columnIndex: int = 0
    paragraphIndex: int = 0
    lines: LineRange = None
    type: Union[Literal["text"], Literal["table"], Literal["figure"]]


class PageJsonFormat(BaseModel):
    pageNumber: Optional[int] = None 
    internalPageNumber: Optional[int] = None
    content: List[PageContentItem] = None


class DocumentJsonFormat(BaseModel):
    documentName: str
    pages: List[PageJsonFormat]






class Annotation(BaseModel):
    annotationText: Optional[str] = None
    documentName: Optional[str] = None
    pageNumber: Optional[int] = None
    internalPageNumber: Optional[int] = None
    columnNumber: Optional[int] = None
    paragraphNumber: Optional[int] = None
    linesInParagraph: Optional[LineRange] = None
    formattedInformation: Optional[str] = None


class ClaimResult(BaseModel):
    claim: str
    annotationDetails: Optional[Annotation] = None


class CustomFilePayload(BaseModel):
    filename: str
    content_type: str
    content: bytes


class TaskDocumentVariants(BaseModel):
    raw_file: CustomFilePayload
    text_file: Optional[Dict[str, Any]]
    text_file_with_metadata: Optional[DocumentJsonFormat]


class ClaimsExtractionTask(BaseModel):
    task_status: TaskStatus
    task_document: Optional[TaskDocumentVariants] = None
    task_uuid: str
    results: Optional[List[ClaimResult]]






class ExtractedImage(BaseModel):
    page_number: int
    image_index: int
    base64_data: str

class PDFExtractionResult(BaseModel):
    full_text: str
    images: List[ExtractedImage]
