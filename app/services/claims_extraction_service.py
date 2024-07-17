import asyncio
import re
from typing import List, Optional
from anthropic import Anthropic
from fastapi import UploadFile
from typing import List, Tuple



from app.pydantic_schemas.claims_extraction_task import (
    Annotation,
    ClaimResult,
    ClaimsExtractionTask,
    ExtractedImage,
    LineRange,
    PDFExtractionResult,
    PageContentItem,
    PageJsonFormat,
    TaskDocumentVariants,
    DocumentJsonFormat,
)
from app.pydantic_schemas.utlis import PDFExtractionResults
from app.services.llm_manager import LLMManager
from app.services.pdf_structure_extractor import PDFStructureExtractor
from app.services.tasks_store import ClaimsExtractionStore
from app.services.universal_file_processor import FileProcessingService
from app.utils.enums import TaskStatus
from app.utils.utils import generate_uuid, normalize_text
from app.config import settings
import fitz  # PyMuPDF
from io import BytesIO
from PIL import Image
import base64

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# from sentence_transformers import SentenceTransformer, util

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# import numpy as np
from typing import Optional, List
from fuzzywuzzy import fuzz




class ClaimsExtractionService:
    def __init__(self, task: ClaimsExtractionTask):
        self.task = task
        self.llm_manager = LLMManager()
        self.anthropic_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.pdf_structure_extractor = PDFStructureExtractor(self.task)

    def run(self):
        #Extract the text_file_with_metadata from the raw_file
        self.pdf_structure_extractor.format_clinical_document()
        doc = fitz.open(stream=self.task.task_document.raw_file.content, filetype="pdf")
        for page_range in self._run_get_page_ranges(doc):
            asyncio.run(self._run_process_page_range(page_range))
            
        self.task.task_status = TaskStatus.COMPLETE


    def _run_get_page_ranges(self, doc: fitz.Document, number_of_selected_pages_per_chunk: int = 3) -> list[tuple[int, int]]:
        total_pages = doc.page_count
        page_ranges = []

        for i in range(0, total_pages, number_of_selected_pages_per_chunk):
            end_page = min(i + number_of_selected_pages_per_chunk, total_pages)
            page_ranges.append((i, end_page))

        return page_ranges


    async def _run_process_page_range(self, page_range: Tuple[int, int]):
        print("Start Run Process for pages", page_range)
        #TODO: Fix this function
        pdf_extraction_results = self.extract_full_text_and_images(*page_range)
        print("Extracted full text and images for pages", page_range)
        claims = await self._run_extract_claims(pdf_extraction_results, page_range)
        print("Length of claims", len(claims))
        filtered_claims = self._run_filter_claims_based_on_section(claims, ["introduction", "methodology", "results"])
        print("Length of filtered claims", len(filtered_claims))
        self._run_map_claims(filtered_claims, page_range)
        
        
    async def _run_extract_claims(self, pdf_extraction_results: PDFExtractionResults, page_range: Tuple[int, int]):
        try:
            return LLMManager.extract_claims_with_claude(
                self.anthropic_client, 
                pdf_extraction_results.full_text, 
                pdf_extraction_results.images
            )
        except Exception as e:
            print(f"Error in claims extraction for pages {page_range}: {e}")
            return []


    def _run_filter_claims_based_on_section(self, claims: List[dict], desired_sections: List[str]) -> List[dict]:
        return [claim for claim in claims if claim.get('Section').lower() in desired_sections]


    def _run_map_claims(self, claims, page_range: Tuple[int, int]):
        for claim in claims:
            claim_annotation_details: Annotation = self.match_claims_to_blocks(
                search_text=claim["statement"],
                page_range=page_range
            )
            if claim_annotation_details and self.are_paragraphs_similar(
                claim["statement"], 
                claim_annotation_details.annotationText
            ):
                self._run_update_claim_annotation(claim, claim_annotation_details)


    def _run_update_claim_annotation(self, claim, claim_annotation_details: Annotation):
        start_line, end_line = self.extract_line_numbers_in_paragraph(
            claim["statement"], 
            claim_annotation_details.annotationText
        )
        claim_annotation_details.linesInParagraph = LineRange(start=start_line, end=end_line)
        claim_annotation_details.formattedInformation += f"/lns {start_line}-{end_line}"
        self.task.results.append(ClaimResult(
            claim=claim["statement"], 
            annotationDetails=claim_annotation_details
        ))

    @staticmethod
    def create_and_store_task(clinical_file: UploadFile) -> ClaimsExtractionTask:
        clinical_file_content = FileProcessingService.extract_file_content(
            clinical_file
        )
        
        file_formatted = TaskDocumentVariants(
            raw_file=clinical_file_content,
            text_file={},
            text_file_with_metadata=DocumentJsonFormat(
                documentName=clinical_file.filename, pages=[]
            ),
        )
        
        claims_extraction_task = ClaimsExtractionTask(
            task_status=TaskStatus.PENDING,
            task_document=file_formatted,
            task_uuid=generate_uuid(),
            results=[],
        )
        
        ClaimsExtractionStore.store_task_in_tracker(claims_extraction_task)
        return claims_extraction_task


    def extract_full_text_and_images(self, start_page: int, end_page: int) -> PDFExtractionResult:
        full_text = ""
        images = []

        try:
            doc = fitz.open(stream=self.task.task_document.raw_file.content, filetype="pdf")

            for page_num in range(start_page, end_page):
                page = doc[page_num]
                full_text += f"Page {page_num + 1}\n" + page.get_text("text") + "\n\n"

                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Convert image bytes to PIL Image - Convert image to base64
                    pil_image = Image.open(BytesIO(image_bytes)) 
                    buffered = BytesIO()
                    image_format = pil_image.format if pil_image.format else "PNG"
                    pil_image.save(buffered, format=image_format)
                    image_base64 = base64.b64encode(buffered.getvalue()).decode()

                    images.append(ExtractedImage(page_number=page_num + 1, image_index=img_index + 1, base64_data=image_base64, image_format=image_format.lower()))

        except Exception as e:
            print(f"Error extracting full text and images: {str(e)}")

        return PDFExtractionResult(full_text=full_text, images=images)



    def match_claims_to_blocks(self, search_text: str, page_range: Optional[List[int]] = None) -> Optional[Annotation]:
        def match_subtext(subtext: str, text_blocks: List[str], threshold: int = 25) -> Optional[Tuple[str, int]]:
            subtext = normalize_text(subtext)
            matched_blocks = [
                (block, fuzz.partial_ratio(subtext, normalize_text(block)))
                for block in text_blocks
                if fuzz.partial_ratio(subtext, normalize_text(block)) >= threshold
            ]
            if matched_blocks:
                matched_blocks.sort(key=lambda x: x[1], reverse=True)
                return matched_blocks[0]
            return None

        def process_content(page_content: List[PageContentItem], page_to_process: PageJsonFormat) -> Optional[Annotation]:
            document_name = self.task.task_document.text_file_with_metadata.documentName
            text_blocks = [content.text for content in page_content]
            top_match = match_subtext(search_text, text_blocks)

            if top_match and top_match[1] > process_content.highest_similarity:
                highest_similarity_content = next(content for content in page_content if content.text == top_match[0])
                print("-------------------")
                print(f"Match found with score {top_match[1]}")
                print(f"Match found with As Text {top_match[0][:10]} : Claim {search_text}")
                print("-------------------")

                process_content.highest_similarity = top_match[1]
                return Annotation(
                    annotationText=highest_similarity_content.text,
                    documentName=document_name,
                    pageNumber=page_to_process.pageNumber,
                    internalPageNumber=page_to_process.internalPageNumber,
                    columnNumber=highest_similarity_content.columnIndex,
                    paragraphNumber=highest_similarity_content.paragraphIndex,
                    formattedInformation=f"{document_name}/p{page_to_process.internalPageNumber}/col{highest_similarity_content.columnIndex}/¶{highest_similarity_content.paragraphIndex}",
                )
            return None

        process_content.highest_similarity = 0
        best_match = None

        pages = self.task.task_document.text_file_with_metadata.pages
        if page_range:
            pages = pages[page_range[0]:page_range[1]]

        for page in pages:
            current_match = process_content(page.content, page)
            if current_match:
                best_match = current_match

        return best_match


    @staticmethod
    def extract_line_numbers_in_paragraph(claim: str, annotation_text: str):
        
        def get_line_number(position, text):
            return text[:position].count('\n') + 1
        
        def handle_edge_case(claim, annotation_text):
            if len(normalize_text(claim)) > len(normalize_text(annotation_text)):
                total_lines = annotation_text.count('\n') + 1
                return 1, total_lines
            return None
        
        def find_phrase_position(phrase, text, from_start=True):
            normalized_phrase = normalize_text(phrase)
            normalized_text = normalize_text(text)
            
            if from_start:
                words = normalized_text.split()
                for i in range(len(words)):
                    substring = ' '.join(words[i:i+len(normalized_phrase.split())])
                    ratio = fuzz.ratio(normalized_phrase, substring)
                    if ratio > 90:
                        position = normalized_text.index(substring)
                        print(f"Match found at position {position} with ratio {ratio}")
                        print(f"Matched text: {text[position:position+50]}...")
                        return position
            else:
                words = normalized_text.split()
                for i in range(len(words)-1, -1, -1):
                    substring = ' '.join(words[max(0, i-len(normalized_phrase.split())+1):i+1])
                    ratio = fuzz.ratio(normalized_phrase, substring)
                    if ratio > 90:
                        position = normalized_text.index(substring) + len(substring)
                        print(f"Match found at position {position} with ratio {ratio}")
                        print(f"Matched text: ...{text[max(0, position-50):position]}")
                        return position
            
            return -1
        
        edge_case_result = handle_edge_case(claim, annotation_text)
        if edge_case_result:
            return edge_case_result
        
        start_pos = find_phrase_position(claim, annotation_text, from_start=True)
        end_pos = find_phrase_position(claim, annotation_text, from_start=False)
        
        print(f"Start Position: {start_pos}, End Position: {end_pos}")
        
        if start_pos == -1 or end_pos == -1:
            return -1, -1
        
        start_line = get_line_number(start_pos, annotation_text)
        end_line = get_line_number(end_pos, annotation_text)
        
        return start_line, end_line
    
        
    @staticmethod
    def are_paragraphs_similar(paragraph1: str, paragraph2: str, threshold: float = 0.5) -> bool:
        """
        Determine if two paragraphs are similar based on cosine similarity.
        
        :param paragraph1: First paragraph text.
        :param paragraph2: Second paragraph text.
        :param threshold: Similarity threshold to determine if paragraphs are similar.
        :return: True if similar, False otherwise.
        """
        paragraph1 = normalize_text(paragraph1)
        paragraph2 = normalize_text(paragraph2)
        
        vectorizer = TfidfVectorizer().fit_transform([paragraph1, paragraph2])
        vectors = vectorizer.toarray()
        
        cosine_sim = cosine_similarity(vectors)
        similarity_score = cosine_sim[0, 1]
        
        return similarity_score >= threshold


