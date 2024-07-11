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
from app.services.universal_file_processor import FileProcessingService
from app.utils.enums import TaskStatus
from app.utils.utils import generate_uuid
from app.config import settings
import fitz  # PyMuPDF
from io import BytesIO
from PIL import Image
from difflib import SequenceMatcher
import base64

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# from sentence_transformers import SentenceTransformer, util


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
            pdf_extraction_results = self.extract_full_text_and_images(i, min(i + chunk_size, total_pages))
            print("Extracting Claims")
            
            try:
                claims = LLMManager.extract_claims_with_claude(self.anthropic_client, pdf_extraction_results.full_text, pdf_extraction_results.images)
            except Exception as e:
                claims = []
                print(f"Error in the claims extraction in this group of pages {i, min(i + chunk_size, total_pages)}")
                print("The error is: ",e)
                
            print("Map Claims")
            for claim in claims:
                # Map the claims by search match to the JSON structure
                claim_annotation_details: Annotation = self.match_claims_to_blocks(search_text=claim["statement"], page_range=[i, min(i + chunk_size, total_pages)])
                if claim_annotation_details is not None and self.are_paragraphs_similar(claim["statement"], claim_annotation_details.annotationText):
                    start_line, end_line = self.extract_line_numbers_in_paragraph(claim["statement"], claim_annotation_details.annotationText)
                    claim_annotation_details.linesInParagraph = LineRange(start=start_line, end=end_line)
                    claim_annotation_details.formattedInformation += f"/lns {claim_annotation_details.linesInParagraph.start}-{claim_annotation_details.linesInParagraph.end}"
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
        def get_similarity(a: str, b: str) -> float:
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()

        def process_content(content_to_process: PageContentItem, page_to_process: PageJsonFormat) -> Optional[Annotation]:
            similarity = get_similarity(content_to_process.text, search_text)
            document_name = self.task.task_document.text_file_with_metadata.documentName
            if similarity > process_content.highest_similarity:
                process_content.highest_similarity = similarity
                return Annotation(
                    annotationText=content_to_process.text,
                    documentName=document_name,
                    pageNumber=page_to_process.pageNumber,
                    internalPageNumber=page_to_process.internalPageNumber,
                    columnNumber=content_to_process.columnIndex,  
                    paragraphNumber=content_to_process.paragraphIndex,  
                    formattedInformation=f"{document_name}/p{page_to_process.internalPageNumber}/col{content_to_process.columnIndex}/¶{content_to_process.paragraphIndex}",
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
            return text[:position].count('\n') +1
        
        # TODO: Remove this edge case function in the future
        def handle_edge_case(claim, annotation_text):
            if len(normalize_text(claim)) > len(normalize_text(annotation_text)):
                total_lines = annotation_text.count('\n') + 1
                return 1, total_lines
            return None


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


        edge_case_result = handle_edge_case(claim, annotation_text)
        if edge_case_result:
            return edge_case_result

        start_pos = find_phrase_position(claim, annotation_text, from_start=True)
        end_pos = find_phrase_position(claim, annotation_text, from_start=False)

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
        vectorizer = TfidfVectorizer().fit_transform([paragraph1, paragraph2])
        vectors = vectorizer.toarray()
        
        cosine_sim = cosine_similarity(vectors)
        similarity_score = cosine_sim[0, 1]
        
        return similarity_score >= threshold

    # @staticmethod
    # def are_sentences_similar(sentence1: str, sentence2: str, threshold: float = 0.5) -> bool:
    #     model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    #     embeddings = model.encode([sentence1, sentence2])
    #     cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        
    #     return cosine_sim.item() >= threshold
