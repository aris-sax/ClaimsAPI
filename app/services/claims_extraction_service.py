from typing import List, Optional
from anthropic import Anthropic
from fastapi import UploadFile

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
from app.services.llm_manager import LLMManager
from app.services.pdf_structure_extractor import PDFStructureExtractor
from app.services.tasks_store import ClaimsExtractionStore
from app.services.universal_file_processor import (
    FileProcessingService,
    ImageTextExtractor,
)
from app.utils.enums import TaskStatus
from app.utils.utils import generate_uuid, preprocess_text
from app.config import settings
import fitz  # PyMuPDF
from io import BytesIO
from PIL import Image
from difflib import SequenceMatcher


class ClaimsExtractionService:
    def __init__(self, task: ClaimsExtractionTask):
        self.task = task
        self.llm_manager = LLMManager()
        self.anthropic_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.pdf_structure_extractor = PDFStructureExtractor(self.task)

    def run(self):
        self.pdf_structure_extractor.format_clinical_document()
        
        doc = fitz.open(stream=self.task.task_document.raw_file.content, filetype="pdf")
        total_pages = len(doc)
        chunk_size = 3
        for i in range(0, total_pages, chunk_size):
            print("Extracting Text and Images")
            full_text, images = self.extract_full_text_and_images(i, min(i + chunk_size, total_pages))
            print("Extracting Claims")
            claims = LLMManager.extract_claims_with_claude(self.anthropic_client, full_text, images)
            print("Map Claims")
            for claim in claims:
                # Map the claims by search match to the JSON structure
                claim_annotation_details: Annotation = self.match_claims_to_blocks(search_text=claim["statement"], page_range=[i, min(i + chunk_size, total_pages)])
                if claim_annotation_details:
                    start_line, end_line = self.extract_line_numbers_in_paragraph(claim["statement"], claim_annotation_details.annotationText) or (0, 0)
                    claim_annotation_details.linesInParagraph = LineRange(start=start_line, end=end_line)
                    self.task.results.append(ClaimResult(claim=claim["statement"], annotationDetails=claim_annotation_details))
                    
        self.task.task_status = TaskStatus.COMPLETE


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

                    # Convert image bytes to PIL Image
                    pil_image = Image.open(BytesIO(image_bytes))

                    # Convert image to base64
                    base64_image = ImageTextExtractor.image_to_base64(pil_image)

                    images.append(ExtractedImage(page_number=page_num + 1, image_index=img_index + 1, base64_data=base64_image))

        except Exception as e:
            print(f"Error extracting full text and images: {str(e)}")

        return full_text, images


    def fuzzy_match_claims_to_blocks(self, claims: List[str], text_blocks: List[PageJsonFormat]) -> List[PageJsonFormat]:
        matched_claims = []
        
        for claim_index, claim in enumerate(claims):
            print(f"\nMatching claim {claim_index + 1}:")
            print(f"Claim: {claim[:100]}...")
            
            best_match = None
            best_ratio = 0
            best_substring_ratio = 0
            preprocessed_claim = preprocess_text(claim)
            
            for block_index, block in enumerate(text_blocks):
                preprocessed_block = preprocess_text(block['text'])
                
                if preprocessed_claim in preprocessed_block:
                    substring_ratio = len(preprocessed_claim) / len(preprocessed_block)
                    if substring_ratio > best_substring_ratio:
                        best_substring_ratio = substring_ratio
                        best_match = block
                
                if best_substring_ratio == 0:
                    ratio = SequenceMatcher(None, preprocessed_claim, preprocessed_block).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = block
            
            if best_match:
                if best_substring_ratio > 0:
                    print(f"Matched as substring (ratio: {best_substring_ratio:.2f})")
                else:
                    print(f"Matched with fuzzy ratio: {best_ratio:.2f}")
                
                matched_claim = claim.copy()
                matched_claim.update({
                    'column': best_match['column'],
                    'paragraph': best_match['paragraph'],
                    'matched_text': best_match['text']
                })
                matched_claims.append(matched_claim)
                print("Matched with block:")
                print(f"  Column {best_match['column']}, Paragraph {best_match['paragraph']}")
                print(f"  {best_match['text'][:200]}...")
            else:
                print(f"Failed to match. Best substring ratio: {best_substring_ratio:.2f}, Best fuzzy ratio: {best_ratio:.2f}")
        
        print(f"\nTotal matched claims: {len(matched_claims)}")

        return matched_claims


    def match_claims_to_blocks(self, search_text: str, page_range: Optional[List[int]] = None) -> Optional[Annotation]:
        def get_similarity(a: str, b: str) -> float:
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()

        def process_content(content: PageContentItem, page: PageJsonFormat) -> Optional[Annotation]:
            similarity = get_similarity(content.text, search_text)
            if similarity > process_content.highest_similarity:
                process_content.highest_similarity = similarity
                return Annotation(
                    annotationText=content.text,
                    documentName=self.task.task_document.text_file_with_metadata.documentName,
                    pageNumber=page.pageNumber,
                    internalPageNumber=page.internalPageNumber,
                    columnNumber=content.columnIndex,  
                    paragraphNumber=content.paragraphIndex,  
                    formattedInformation=f"Similarity: {similarity:.2f}"
                )
            return None

        process_content.highest_similarity = 0
        best_match = None

        pages = self.task.task_document.text_file_with_metadata.pages
        if page_range:
            pages = pages[page_range[0]:page_range[1]]

        for page in pages:
            for content in page.content:
                current_match = process_content(content, page)
                if current_match:
                    best_match = current_match

        return best_match


    @staticmethod
    def extract_line_numbers_in_paragraph(claim: str, annotation_text: str):
        
        def normalize_text(text):
            return ' '.join(text.split())
        
        def get_line_number(position, text):
            return text[:position].count('\n')
        
        def find_phrase_position(phrase, text, from_start=True):
            normalized_phrase = normalize_text(phrase)
            normalized_text = normalize_text(text)
            find_func = normalized_text.find if from_start else normalized_text.rfind
            
            position = find_func(normalized_phrase)
            if position != -1:
                return position + (0 if from_start else len(normalized_phrase))
            
            # If exact match not found, try partial matches
            words = normalized_phrase.split()
            for i in range(len(words), 0, -1):
                partial_phrase = ' '.join(words[:i] if from_start else words[-i:])
                position = find_func(partial_phrase)
                if position != -1:
                    return position + (0 if from_start else len(partial_phrase))
            
            return -1

        start_pos = find_phrase_position(claim, annotation_text, from_start=True)
        end_pos = find_phrase_position(claim, annotation_text, from_start=False)

        if start_pos == -1 or end_pos == -1:
            return None, None

        start_line = get_line_number(start_pos, annotation_text)
        end_line = get_line_number(end_pos, annotation_text)

        return start_line, end_line

