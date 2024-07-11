import base64
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import BinaryIO, List

import fitz
import docx
from PIL import Image, ImageFile, UnidentifiedImageError
from fastapi import UploadFile

from app.pydantic_schemas.utlis import ImageTracker
from app.pydantic_schemas.claims_extraction_task import CustomFilePayload
from app.services.llm_manager import LLMManager
from app.utils.enums import FileType
from app.utils.utils import clean_string

ImageFile.LOAD_TRUNCATED_IMAGES = True


class UniversalFileProcessor:
    def __init__(self):
        self.extractors = {
            FileType.PDF: PDFTextExtractor(),
            FileType.IMAGE: ImageTextExtractor(),
        }

    def convert_file_to_text(self, file: BinaryIO, file_type: FileType) -> str:
        extractor = self.extractors.get(file_type)
        if not extractor:
            raise ValueError(f"Unsupported file type: {file_type}")
        cleaned_data = clean_string(extractor.extract_text(file))
        return cleaned_data


class PDFTextExtractor:
    def __init__(self):
        self.llm_manager = LLMManager()

    def extract_text(self, file: BinaryIO) -> str:
        text = []
        try:
            images_tracker = self.extract_images_from_pdf(file)
            images_tracker_with_text = self.process_images_in_parallel(images_tracker)
            images_text = [image_tracker.image_text for image_tracker in images_tracker_with_text]
            text.extend(images_text)

            file.seek(0)  # Reset file pointer to the beginning before reading
            pdf_document = fitz.open(stream=file.read(), filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text.append(page.get_text())

            if text and " ".join(text).strip() == "":
                raise ValueError("PDF file is empty")

            return "\n".join(text)

        except ValueError:
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise  # Re-raise any other exceptions to stop the app

    def process_images_in_parallel(self, image_trackers: List[ImageTracker]) -> List[ImageTracker]:
        def process_image(image_tracker: ImageTracker):
            image_text = self.convert_image_to_text(image_tracker.image)
            image_tracker.image_text = image_text
            return image_tracker

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_image, image_tracker): image_tracker for image_tracker in image_trackers}
            for future in as_completed(futures):
                image_tracker = futures[future]
                try:
                    image_tracker = future.result()
                except Exception as e:
                    print(f"Error processing image on page {image_tracker.page_number}: {e}")
        return image_trackers

    @staticmethod
    def extract_images_from_pdf(file: BinaryIO) -> List[ImageTracker]:
        images_tracker: List[ImageTracker] = []
        file.seek(0)  # Reset file pointer to the beginning before reading
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", Image.DecompressionBombWarning)
                        img: Image = Image.open(BytesIO(image_bytes))
                        img.verify()  # Verify the integrity of the image
                        img = Image.open(BytesIO(image_bytes))  # Re-open the image to reset the file pointer
                        images_tracker.append(ImageTracker(image=img, page_number=page_num))
                except (UnidentifiedImageError, Image.DecompressionBombWarning) as e:
                    print(f"Unidentified image at page {page_num + 1}, image index {img_index}: {e}")
                    continue  # Skip invalid images
                except Exception as e:
                    print(f"Error processing image at page {page_num + 1}, image index {img_index}: {e}")
                    continue  # Skip other errors
        return images_tracker

    def convert_image_to_text(self, image: Image.Image) -> str:
        img_encoded_str = self.image_to_base64(image)
        model_response = self.llm_manager.convert_image_to_text_using_vlm(img_encoded_str)
        json_model_response = json.loads(model_response)
        image_text = ""
        if json_model_response["isImageContainText"]:
            image_text = json_model_response["text"]
        return image_text

    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        buffered = BytesIO()
        image_format = image.format if image.format else "PNG"
        if image_format.lower() not in ["jpeg", "jpg", "png"]:
            image_format = "JPEG"
        image.save(buffered, format=image_format)
        return f"data:image/{image_format.lower()};base64," + base64.b64encode(buffered.getvalue()).decode()

    @staticmethod
    def pdf_page_to_base64(page) -> str:
        # Convert the page to a Pixmap
        pix = page.get_pixmap()

        # Convert the Pixmap to an Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Convert the Image to a BytesIO object
        buffered = BytesIO()
        image_format = img.format if img.format else "PNG"
        img.save(buffered, format=image_format)

        # Encode the image to base64 and return string with data URI scheme
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{image_format.lower()};base64," + img_base64


class ImageTextExtractor:
    def __init__(self):
        self.llm_manager = LLMManager()

    def extract_text(self, file: BinaryIO) -> str:
        image = Image.open(file)
        return self.convert_image_to_text(image)

    def convert_image_to_text(self, image: Image.Image) -> str:
        img_encoded_str = self.image_to_base64(image)
        model_response = self.llm_manager.convert_image_to_text_using_vlm(img_encoded_str)
        json_model_response = json.loads(model_response)
        image_text = ""
        if json_model_response["isImageContainText"]:
            image_text = json_model_response["text"]
        return image_text

    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        buffered = BytesIO()
        image_format = image.format if image.format else "PNG"
        image.save(buffered, format=image_format)
        return f"data:image/{image_format.lower()};base64," + base64.b64encode(buffered.getvalue()).decode()


class FileProcessingService:
    def __init__(self):
        pass

    @staticmethod
    def extract_file_content(file: UploadFile) -> CustomFilePayload:
        return CustomFilePayload(
                filename=file.filename,
                content_type=file.content_type,
                content=file.file.read()
            )

