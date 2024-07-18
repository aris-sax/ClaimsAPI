import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple
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
from app.pydantic_schemas.utlis import PDFExtractionResults
from app.services.llm_manager import LLMManager
from app.services.pdf_structure_extractor import PDFStructureExtractor
from app.services.tasks_store import ClaimsExtractionStore
from app.services.universal_file_processor import FileProcessingService
from app.utils.enums import TaskStatus
from app.utils.utils import combined_best_match, generate_uuid, normalize_text
from app.config import settings
import fitz  # PyMuPDF
from io import BytesIO
from PIL import Image
import base64

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        page_ranges = self._run_get_page_ranges(doc)
        print("Page ranges:", page_ranges)
        # Run the concurrent processing of page ranges
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(self._run_process_page_range, page_range) for page_range in page_ranges]
            for future in futures:
                result = future.result()
                # print("Processed pages result:", result)

        self.task.task_status = TaskStatus.COMPLETE


    def _run_get_page_ranges(self, doc: fitz.Document, number_of_selected_pages_per_chunk: int = 3) -> list[tuple[int, int]]:
        total_pages = doc.page_count
        page_ranges = []

        for i in range(0, total_pages, number_of_selected_pages_per_chunk):
            end_page = min(i + number_of_selected_pages_per_chunk, total_pages)
            page_ranges.append((i, end_page))

        return page_ranges


    def _run_process_page_range(self, page_range: Tuple[int, int]):
        print("Start Run Process for pages", page_range)
        #TODO: Fix this function
        pdf_extraction_results = self.extract_full_text_and_images(*page_range)
        claims = asyncio.run(self._run_extract_claims(pdf_extraction_results, page_range))
        print("Length of claims", len(claims))
        filtered_claims = self._run_filter_claims_based_on_section(claims, ["introduction", "methodology", "results"])
        print("Length of filtered claims", len(filtered_claims))
        self._run_map_claims(filtered_claims, page_range)
        return f"Processed pages {page_range}"
        
        
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
            claim_annotation_details: Annotation | None = self.match_claims_to_blocks(
                search_text=claim["statement"],
                page_range=page_range
            )
            if claim_annotation_details:
                self._run_update_claim_annotation(claim, claim_annotation_details)
            else:
                print(f"Claim not found in the document: {claim['statement']}")
                # self.task.results.append(ClaimResult(claim=claim["statement"]))


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



    def match_claims_to_blocks_old(self, search_text: str, page_range: Optional[List[int]] = None) -> Optional[Annotation]:

        def match_subtext(subtext: str, text_blocks: List[str], threshold: float = 0.1) -> Optional[Tuple[int, float]]:
            if not text_blocks:
                print("ERROR: text_blocks list is empty")

            all_texts = [normalize_text(text) for text in text_blocks]

            # Add the normalized subtext to the list of texts
            subtext_normalized = normalize_text(subtext)
            all_texts.append(subtext_normalized)

            # Compute TF-IDF vectors for all texts
            vectorizer = TfidfVectorizer().fit_transform(all_texts)
            vectors = vectorizer.toarray()

            if vectors.shape[0] < 2:
                raise ValueError("Not enough text data to compute similarities")

            # Calculate cosine similarity between the subtext and all document texts
            similarities = cosine_similarity([vectors[-1]], vectors[:-1])[0]

            # Find the most similar text
            most_similar_index = similarities.argmax()
            similarity_score = similarities[most_similar_index]

            if similarity_score > threshold:
                return most_similar_index, similarity_score
            else:
                return None

        def process_content(page_content: List[PageContentItem], page_to_process: PageJsonFormat) -> Optional[Annotation]:
            document_name = self.task.task_document.text_file_with_metadata.documentName

            if not page_content:
                print(f"Page content is empty for page {page_to_process.pageNumber}")
                return None

            text_blocks = [content.text for content in page_content]

            if not text_blocks:
                print(f"Text blocks are empty for page {page_to_process.pageNumber}")
                return None

            matched_block = match_subtext(search_text, text_blocks)

            if matched_block and matched_block[1] > process_content.highest_similarity:
                highest_similarity_content = page_content[matched_block[0]]

                process_content.highest_similarity = matched_block[1]
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

        process_content.highest_similarity = 0  # Reset for each call
        best_match = None

        if page_range is None:
            raise ValueError("page_range cannot be None")

        start_page, end_page = page_range
        pages = self.task.task_document.text_file_with_metadata.pages[start_page:end_page]

        for page in pages:
            current_match = process_content(page.content, page)
            if current_match and current_match.pageNumber >= start_page and current_match.pageNumber < end_page:
                best_match = current_match

        return best_match


    def match_claims_to_blocks(self, search_text: str, page_range: Optional[List[int]] = None) -> Optional[Annotation]:
        
        highest_match_similarity = 0  # Reset for each call
        best_match_block = None

        if page_range is None:
            raise ValueError("page_range cannot be None")

        start_page, end_page = page_range
        pages = self.task.task_document.text_file_with_metadata.pages[start_page:end_page]
        

        for page in pages:
            document_name = self.task.task_document.text_file_with_metadata.documentName
            page_content = page.content

            if not page_content:
                print(f"Page content is empty for page {page.pageNumber}")
                continue

            text_blocks = [content.text for content in page_content]
            text_blocks_normalized = [normalize_text(text) for text in text_blocks]
            search_text_normalized = normalize_text(search_text)

            if not text_blocks:
                print(f"Text blocks are empty for page {page.pageNumber}")
                continue

            best_match_content_and_similarity = combined_best_match(search_text_normalized, text_blocks_normalized)

            if best_match_content_and_similarity and best_match_content_and_similarity[1] > highest_match_similarity:
                best_match_index = best_match_content_and_similarity[0]
                highest_similarity_content = page_content[best_match_index]

                highest_match_similarity = best_match_content_and_similarity[1]
                best_match_block = Annotation(
                    annotationText=highest_similarity_content.text,
                    documentName=document_name,
                    pageNumber=page.pageNumber,
                    internalPageNumber=page.internalPageNumber,
                    columnNumber=highest_similarity_content.columnIndex,
                    paragraphNumber=highest_similarity_content.paragraphIndex,
                    formattedInformation=f"{document_name}/p{page.internalPageNumber}/col{highest_similarity_content.columnIndex}/¶{highest_similarity_content.paragraphIndex}",
                )

        return best_match_block

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


