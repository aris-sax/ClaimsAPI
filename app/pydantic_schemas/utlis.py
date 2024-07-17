
from typing import List
from PIL.Image import Image
from pydantic import BaseModel


class ImageTracker(BaseModel):
    image: Image
    page_number: int
    image_text: str = None

    class Config:
        arbitrary_types_allowed = True



class PDFExtractionResults(BaseModel):
    full_text: str
    images: List[ImageTracker]  # Assuming images are represented as base64 strings
