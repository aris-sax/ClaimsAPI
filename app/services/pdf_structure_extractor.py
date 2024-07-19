import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher
import fitz  # PyMuPDF
from fitz import Page
import numpy as np
from anthropic import Anthropic
from PIL import Image
from app.pydantic_schemas.claims_extraction_task import ClaimsExtractionTask, DocumentJsonFormat, PageContentItem, PageJsonFormat
from app.services.layout_ai_model import Detectron2LayoutModelSingleton
from app.services.llm_manager import LLMManager
from app.services.universal_file_processor import PDFTextExtractor
from app.config import settings

class PDFStructureExtractor:
    """
    A class to extract structured text blocks from a PDF document.
    """

    def __init__(self, task: ClaimsExtractionTask):
        self.task = task
        self.llm_manager_anthropic = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.llm_manager = LLMManager()
        self.document_text_blocks = []
        self.logger = logging.getLogger(__name__)

    def format_clinical_document(self):
        with fitz.open(stream=self.task.task_document.raw_file.content, filetype="pdf") as doc:
            print("Extracting internal page numbering...")
            numbering_start_page, first_internal_number = self._extract_internal_page_numbering(doc)

            document_name = self.task.task_document.raw_file.filename
            self.task.task_document.text_file_with_metadata = DocumentJsonFormat(documentName=document_name, pages=[])

            print("Processing pages...")
            self.process_pages_async_cover(doc, numbering_start_page, first_internal_number)

        return self.task.task_document.text_file_with_metadata

    def process_pages_async_cover(self, doc, numbering_start_page, first_internal_number):
        self._process_pages_async(doc, numbering_start_page, first_internal_number)

    def _process_pages_async(self, doc, numbering_start_page, first_internal_number):
        MAX_WORKERS = 5
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(self._process_page_data_async, doc, page_num, numbering_start_page, first_internal_number)
                for page_num in range(len(doc))
            ]
            for future in as_completed(futures):
                page_formatted = future.result()
                self.task.task_document.text_file_with_metadata.pages.append(page_formatted)
        #After all tasks are completed arrange the pages data based on the page_formatted.pageNumber  value
        self.task.task_document.text_file_with_metadata.pages.sort(key=lambda x: x.pageNumber)


    def _process_page_data_async(self, doc, page_num, numbering_start_page, first_internal_number):
        current_page_internal_page_number = self._calculate_internal_page_number(page_num, numbering_start_page, first_internal_number)

        page = doc.load_page(page_num)
        model = Detectron2LayoutModelSingleton.get_instance()
        page_as_blocks = self.process_page_sync(page, page_num, model, self.llm_manager_anthropic)

        page_blocks_formatted = [self._format_block(block) for block in page_as_blocks]

        return PageJsonFormat(
            pageNumber=page_num + 1,
            internalPageNumber=current_page_internal_page_number,
            content=page_blocks_formatted,
        )

    def _process_pages(self, doc, numbering_start_page, first_internal_number):
        MAX_WORKERS = 5
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(self._process_page_data, doc, page_num, numbering_start_page, first_internal_number)
                for page_num in range(len(doc))
            ]
            for future in as_completed(futures):
                page_formatted = future.result()
                self.task.task_document.text_file_with_metadata.pages.append(page_formatted)

    def _process_page_data(self, doc, page_num, numbering_start_page, first_internal_number):
        current_page_internal_page_number = self._calculate_internal_page_number(page_num, numbering_start_page, first_internal_number)

        page = doc.load_page(page_num)
        model = Detectron2LayoutModelSingleton.get_instance()
        page_as_blocks = self.process_page_sync(page, page_num, model, self.llm_manager_anthropic)

        page_blocks_formatted = [self._format_block(block) for block in page_as_blocks]

        return PageJsonFormat(
            pageNumber=page_num + 1,
            internalPageNumber=current_page_internal_page_number,
            content=page_blocks_formatted,
        )

    @staticmethod
    def _calculate_internal_page_number(page_num, numbering_start_page, first_internal_number):
        if page_num < numbering_start_page:
            return page_num + (first_internal_number - numbering_start_page) 
        return first_internal_number + (page_num - numbering_start_page)

    @staticmethod
    def _format_block(block):
        return PageContentItem(
            text=block["text"],
            paragraphIndex=block["paragraph"],
            columnIndex=block["column"],
            type="text"
        )

    def _extract_internal_page_numbering(
        self, pdf_document: fitz.Document, max_pages_to_check: int = 5
    ) -> Tuple[int, int]:
        num_pages = len(pdf_document)
        pages_to_check = min(max_pages_to_check, num_pages)
        
        def extract_number(page):
            page_as_image = PDFTextExtractor.pdf_page_to_base64(page)
            return self.llm_manager.extract_page_number_from_image(page_as_image)


        async def extract_numbers_async(current_page_num):
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                current_page = pdf_document.load_page(current_page_num)
                next_page = pdf_document.load_page(current_page_num + 1)
                
                extracted_number, next_page_extracted_number = await asyncio.gather(
                    loop.run_in_executor(pool, extract_number, current_page),
                    loop.run_in_executor(pool, extract_number, next_page)
                )
                
                return extracted_number, next_page_extracted_number

        for current_page_num in range(pages_to_check):
            extracted_number, next_page_extracted_number = asyncio.run(extract_numbers_async(current_page_num))
            
            if extracted_number != 0 and extracted_number == (next_page_extracted_number - 1):
                return current_page_num, extracted_number

        return 0, 0

    def process_page(self, page: Page, page_num: int, model: any, client: Anthropic) -> List[Dict[str, Any]]:
        print(f"Processing page {page_num + 1}")

        pix = page.get_pixmap()

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        np_image = np.array(img)

        np_image = self.normalize_image(np_image)
        layout = model.detect(np_image)

        filtered_blocks = self.filter_blocks(page, layout)
        self.logger.info(f"Detected {len(filtered_blocks)} text blocks on page {page_num + 1}")

        page_texts = self.extract_page_texts(page, pix, filtered_blocks)
        deduplicated_texts = self.deduplicate_texts(page_texts)
        text_blocks = self.append_text_blocks(deduplicated_texts, client)
        print(f"Page {page_num + 1} processed successfully. Text blocks extracted: {len(text_blocks)}")
        return text_blocks

    def process_page_sync(self, page: Page, page_num: int, model: any, client: Anthropic) -> List[Dict[str, Any]]:
        return self.process_page(page, page_num, model, client)

    def append_text_blocks(self, deduplicated_texts: List[Dict[str, Any]], client: Anthropic) -> List[Dict[str, Any]]:
        text_blocks = []
        columns = {1: [], 2: []}
        for text in deduplicated_texts:
            columns[text["column"]].append(text)

        for column in columns.values():
            column.sort(key=lambda t: t["y"])

        for column, texts in enumerate(columns.values(), 1):
            paragraph_number = 1
            for text in texts:
                if LLMManager.classify_text_with_claude(client, text['text']):
                    text_blocks.append({
                        "column": column,
                        "paragraph": paragraph_number,
                        "text": text['text']
                    })
                    self.logger.info(f"Extracted text block on column {column}, paragraph {paragraph_number}. Length: {len(text['text'])}")
                    paragraph_number += 1
        return text_blocks

    @staticmethod
    def normalize_image(np_image: np.ndarray) -> np.ndarray:
        if np_image.dtype != np.uint8:
            np_image = (np_image * 255).astype(np.uint8)
        if np_image.ndim == 2:
            np_image = np.stack((np_image,) * 3, axis=-1)
        return np_image

    @staticmethod
    def filter_blocks(page: Page, layout: Any) -> List[Any]:
        type_2_blocks = [b for b in layout if b.type == 2 and b.score > 0.5]
        type_0_blocks = [b for b in layout if b.type == 0 and b.score > 0.5]

        filtered_blocks = type_2_blocks.copy()
        for block_0 in type_0_blocks:
            block_0_text = page.get_text("text", clip=(block_0.block.x_1, block_0.block.y_1, block_0.block.x_2, block_0.block.y_2)).strip()
            if not any(
                PDFStructureExtractor.is_subblock(block_0, block_2) or 
                PDFStructureExtractor.similar(
                    block_0_text, 
                    page.get_text("text", clip=(block_2.block.x_1, block_2.block.y_1, block_2.block.x_2, block_2.block.y_2)).strip()
                )
                for block_2 in type_2_blocks
            ):
                filtered_blocks.append(block_0)

        return sorted(filtered_blocks, key=lambda b: (b.block.y_1, b.block.x_1))

    @staticmethod
    def extract_page_texts(page: Page, pix: Any, filtered_blocks: List[Any]) -> List[Dict[str, Any]]:
        page_texts = []
        for block in filtered_blocks:
            block_coords = (block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2)
            block_text = page.get_text("text", clip=block_coords).strip()

            if not block_text or len(block_text) < 50:
                continue

            column = 1 if block.block.x_1 < pix.width * 2/5 else 2
            
            page_texts.append({
                "column": column,
                "text": block_text,
                "y": block.block.y_1,
            })
        return page_texts

    @staticmethod
    def deduplicate_texts(page_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduplicated_texts = []
        for text in page_texts:
            if not any(PDFStructureExtractor.similar(text['text'], t['text']) for t in deduplicated_texts):
                deduplicated_texts.append(text)
        return deduplicated_texts
    
    @staticmethod
    def similar(a: str, b: str) -> bool:
        return SequenceMatcher(None, a, b).ratio() > 0.8

    @staticmethod
    def is_subblock(block1: Any, block2: Any, tolerance: int = 5) -> bool:
        return (block1.block.x_1 >= block2.block.x_1 - tolerance and
                block1.block.y_1 >= block2.block.y_1 - tolerance and
                block1.block.x_2 <= block2.block.x_2 + tolerance and
                block1.block.y_2 <= block2.block.y_2 + tolerance)
