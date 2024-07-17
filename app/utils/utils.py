import os
import time
import uuid
import re

from fastapi import UploadFile, HTTPException

from app.utils.enums import FileType


def generate_uuid():
    base_uuid = str(uuid.uuid4())
    return base_uuid


def retry_operation(operation, retries, delay, *args, **kwargs):
    for attempt in range(retries):
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            print(f"Error in {operation.__name__} (attempt {attempt + 1} of {retries}): ", e)
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise


async def retry_operation_async(operation, retries, delay, *args, **kwargs):
    for attempt in range(retries):
        try:
            return await operation(*args, **kwargs)
        except Exception as e:
            print(f"Error in {operation.__name__} (attempt {attempt + 1} of {retries}): ", e)
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise


def clean_string(input_string):
    # Replace multiple tabs with a single space
    cleaned_string = re.sub(r'\t+', ' ', input_string)
    # Replace multiple newlines with a single newline
    cleaned_string = re.sub(r'\n+', '\n', cleaned_string)
    return cleaned_string.strip()


# Mapping MIME types to FileType
mime_type_to_file_type = {
    "application/pdf": FileType.PDF,
    "image/jpeg": FileType.IMAGE,
    "image/jpg": FileType.IMAGE,
    "image/png": FileType.IMAGE,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": FileType.DOCX,
    "text/plain": FileType.TXT,
}


def validate_file(file: UploadFile, allowed_extensions: set):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}. Only PDF, Word, and TXT "
                                                    f"files are allowed For Clinical files.")


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text



def normalize_text(text):
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"≥", ">=", text)
    text = re.sub(r"≤", "<=", text)
    text = re.sub(r"<", "<", text)
    text = re.sub(r">", ">", text)
    text = text.lower()
    return ' '.join(text.split())
