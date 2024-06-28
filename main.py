import base64
import json
import os
import uuid
import numpy as np
from typing import List, Dict, Any, Tuple
from io import BytesIO
from difflib import SequenceMatcher
import asyncio
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from anthropic import Anthropic
import re

import pytesseract
import fitz  # PyMuPDF
from PIL import Image
import pdfplumber
import layoutparser as lp
import re

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio() > 0.8

# Create directories to save uploaded files and extracted images
UPLOAD_FOLDER = 'uploads'
EXTRACTED_IMAGES_FOLDER = 'extracted_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXTRACTED_IMAGES_FOLDER, exist_ok=True)

home_dir = os.path.expanduser("~")

# Initialize the model with the explicit path
model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config')

# Store task results
task_results = {}

# It's better to use an environment variable for the API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def is_article_text(text: str) -> bool:
    patterns = [
        r"^\d+\.\s", r"^[A-Z][a-z]+ et al\.", r"^Figure \d+", r"^FIG+", r"MD",
        r"^Table \d+", r"^\d+$", r"^[A-Z\s]+$", r"©", r"™", r"Creative Commons"
    ]
    return not any(re.match(pattern, text.strip()) for pattern in patterns) and len(text.strip()) >= 5

def classify_text_with_claude(client: Anthropic, text: str) -> bool:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Imagine you are an annotator trying to identify the paragraph number for each claim in this scientific paper. In order to do this accurately, please classify the following text block as either part of the core scientific article text (true) or not (false) so that you can annotate correctly . All titles, footnotes, author names, miscellaneous other information about the paper should be false, only text that is a part of the paper itself and its content should be true this include purpose, metholds, results, etc..Respond with only 'true' or 'false'.

Text to classify:
{text}

Is this part of the scientific article text?"""
                }
            ]
        }
    ]
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1,
        temperature=0,
        messages=messages
    ).content[0].text.strip().lower()
    
    return response == 'true'

def is_subblock(block1, block2, tolerance=5):
    return (block1.block.x_1 >= block2.block.x_1 - tolerance and
            block1.block.y_1 >= block2.block.y_1 - tolerance and
            block1.block.x_2 <= block2.block.x_2 + tolerance and
            block1.block.y_2 <= block2.block.y_2 + tolerance)

async def process_page(page, page_num, model, doc, client, text_blocks):
    print(f"Processing page {page_num + 1}")

    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    np_image = np.array(img)

    if np_image.dtype != np.uint8:
        np_image = (np_image * 255).astype(np.uint8)
    if np_image.ndim == 2:
        np_image = np.stack((np_image,) * 3, axis=-1)

    layout = await asyncio.to_thread(model.detect, np_image)

    type_2_blocks = [b for b in layout if b.type == 2 and b.score > 0.5]
    type_0_blocks = [b for b in layout if b.type == 0 and b.score > 0.5]

    filtered_blocks = type_2_blocks.copy()
    for block_0 in type_0_blocks:
        block_0_text = page.get_text("text", clip=(block_0.block.x_1, block_0.block.y_1, block_0.block.x_2, block_0.block.y_2)).strip()
        if not any(is_subblock(block_0, block_2) or 
                    similar(block_0_text, page.get_text("text", clip=(block_2.block.x_1, block_2.block.y_1, block_2.block.x_2, block_2.block.y_2)).strip())
                    for block_2 in type_2_blocks):
            filtered_blocks.append(block_0)

    layout_blocks = sorted(filtered_blocks, key=lambda b: (b.block.y_1, b.block.x_1))

    print(f"Detected {len(layout_blocks)} text blocks on page {page_num + 1}")

    page_texts = []

    for block in layout_blocks:
        block_coords = (block.block.x_1, block.block.y_1, block.block.x_2, block.block.y_2)
        block_text = page.get_text("text", clip=block_coords).strip()

        if not block_text or len(block_text) < 50:
            continue

        column = 1 if block.block.x_1 < pix.width / 2 else 2
        
        page_texts.append({
            "column": column,
            "text": block_text,
            "y": block.block.y_1,
        })

    deduplicated_texts = []
    for text in page_texts:
        if not any(similar(text['text'], t['text']) for t in deduplicated_texts):
            deduplicated_texts.append(text)

    columns = {1: [], 2: []}
    for text in deduplicated_texts:
        columns[text["column"]].append(text)
    
    for column in columns.values():
        column.sort(key=lambda t: t["y"])

    for column, texts in enumerate(columns.values(), 1):
        paragraph_number = 1
        for text in texts:
            if classify_text_with_claude(client, text['text']):
                text_blocks.append({
                    "column": column,
                    "paragraph": paragraph_number,
                    "text": text['text']
                })
                print(f"Extracted text block on column {column}, paragraph {paragraph_number}. Length: {len(text['text'])}")
                paragraph_number += 1

async def validate_and_correct_claims(client: Anthropic, extracted_claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Process claims in batches of 10
    batch_size = 10
    all_processed_claims = []

    for i in range(0, len(extracted_claims), batch_size):
        batch = extracted_claims[i:i+batch_size]
        
        messages = [
            {
                "role": "user",
                "content": f"""Please verify and correct the following claims from a scientific paper:

1. Ensure citation format is consistent: FirstAuthor et al. Journal Name Volume(Issue):PageRange
2. Page numbers should match the journal's footer (usually not between 1-20)
3. Correct any inconsistencies in citation, page number, column, paragraph, and line range

Claims to process:
{json.dumps(batch, indent=2)}

Respond ONLY with a JSON object in this exact format, ensuring it's complete and properly closed:
{{
  "processedClaims": [
    {{
      "statement": "Exact claim text",
      "annotation": "FirstAuthorName et al., year, p[Page]/col[Column]/para[Paragraph]/lns[Lines]",
      "citation": "FirstAuthor et al. Journal Name Volume(Issue):PageRange"
    }},
    // ... more claim objects
  ]
}}
"""
            }
        ]
        
        try:
            # Updated API call to match the new Anthropic client requirements
            completion = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0.1,
                messages=messages
            )
            
            # Extract the content from the completion
            response_content = completion.content[0].text
            
            parsed_json = json.loads(response_content)
            if 'processedClaims' in parsed_json:
                all_processed_claims.extend(parsed_json['processedClaims'])
            else:
                print(f"Warning: 'processedClaims' not found in parsed JSON. Raw response: {response_content}")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Raw completion: {response_content}")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    return all_processed_claims
async def extract_text_blocks(pdf_path: str, client: Anthropic, start_page: int, end_page: int) -> List[Dict[str, Any]]:
    text_blocks = []

    try:
        print(f"Loading PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        print(f"PDF opened successfully. Total pages: {len(doc)}")

        model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                         extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5])

        tasks = []
        for page_num in range(start_page, end_page):
            page = doc[page_num]
            tasks.append(process_page(page, page_num, model, doc, client, text_blocks))

        await asyncio.gather(*tasks)

    except Exception as e:
        print(f"Error processing PDF: {e}")

    return text_blocks

async def extract_full_text_and_images(pdf_path: str, start_page: int, end_page: int) -> Tuple[str, List[Tuple[int, int, str]]]:
    full_text = ""
    images = []

    try:
        doc = fitz.open(pdf_path)
        for page_num in range(start_page, end_page):
            page = doc[page_num]
            full_text += f"Page {page_num + 1}\n" + page.get_text("text") + "\n\n"

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                image_filename = f"image_p{page_num + 1}_i{img_index + 1}.png"
                image_path = os.path.join(EXTRACTED_IMAGES_FOLDER, image_filename)
                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)
                
                images.append((page_num + 1, img_index + 1, image_path))

    except Exception as e:
        print(f"Error extracting full text and images: {str(e)}")

    return full_text, images

def get_completion(client: Anthropic, messages: List[Dict[str, Any]]) -> str:
    return client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=2000,
        temperature=0,
        messages=messages
    ).content[0].text

def escape_json_string(s: str) -> str:
    return s.replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')

async def extract_claims(client: Anthropic, full_text: str, images: List[Tuple[int, int, str]]) -> List[Dict[str, Any]]:
    content = [
        {
            "type": "text",
            "text": f"Here is the text of the academic paper:\n\n{full_text}\n\nNow I will provide the images from the paper, if any were successfully extracted."
        }
    ]

    for page_num, img_num, img_path in images:
        try:
            encoded_image = image_to_base64(img_path)
            content.append({
                "type": "image",
                "image_url": f"data:image/png;base64,{encoded_image}"
            })
            content.append({
                "type": "text",
                "text": f"This is image {img_num} from page {page_num} of the paper."
            })
        except Exception as e:
            print(f"Error processing image {img_num} from page {page_num}: {str(e)}")

    content.append({
        "type": "text",
        "text": """You are a highly capable AI assistant tasked with extracting and organizing information from a scientific paper about a drug to then be used on the drug's website. Follow these instructions carefully:

1. CLAIM EXTRACTION:
   - Identify all claims related to: 
     a. Study design 
     b. Patient outcomes and primary and secondary endpoints
     c. Efficacy of drug in treating a specific disease compared to control. Common efficacy metrics include progression free survival (pfs), overall survival (os), objective response rate (ORR), reduction in risk of death, etc.  
     d. Adverse events associated with drug 
   - Try to include less information from the summaries on the first page but instead look for where the data is actually discussed in-depth in the paper
   - Include claims ranging from phrases to full paragraphs or tables
   - Don't include more than 3-4 claims maximum unless needed
   - Focus on extracting claims that are similar in style and content to the following examples:

2. SOURCE IDENTIFICATION:
   - For each claim, note:
     - Page number (use original document footer numbers)
     - Citation in the format: "FirstAuthor et al. Journal Name Volume(Issue):PageRange"

3. JSON OUTPUT STRUCTURE:
   Create a JSON object with the following structure:
   { "extractedClaims": [
       {
         "statement": "Exact claim text",
         "page": "Page number as listed in the document",
         "citation": "FirstAuthor et al. Journal Name Volume(Issue):PageRange"
       },
       // ... more claim objects
     ]
   }

4. SPECIAL CASES:
   - Multi-page claims: Indicate full range in page field
   - Missing info: Use null for missing fields

5. JSON FORMATTING:
   - Ensure valid JSON format
   - Use double quotes for strings
   - Format JSON for readability with appropriate indentation

6. PROCESSING INSTRUCTIONS:
   - Analyze the entire document before starting output
   - Prioritize accuracy over speed
   - If uncertain about a claim's relevance, include it
   - Do not summarize or paraphrase claims; use exact text
   - Try to combine claims that are near each and about the same topic when possible without reducing quality. 

7. SELF-CHECKING:
   - Verify all extracted information meets specified criteria
   - Make sure each claim is relevant to demonstrating the drug's efficacy, adverse events associated with the drug, or study design that would be relevant to a patient or physician interested in the drug. If it is not, then remove the entry from the JSON.
   - Double-check page numbers for accuracy
   - Ensure JSON is well-formed and valid
   - Make sure all citations are consistent

Begin your output with the JSON object as specified in step 3. Do not include any text before or after the JSON output."""
    })

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    completion = get_completion(client, messages)
    try:
        parsed_json = json.loads(completion)
        if 'extractedClaims' not in parsed_json:
            print(f"Warning: 'extractedClaims' not found in parsed JSON. Raw response: {completion}")
            return []
        claims = parsed_json['extractedClaims']
        print(f"Extracted {len(claims)} claims")
        for i, claim in enumerate(claims[:3]):
            print(f"Claim {i + 1}: {claim['statement'][:100]}...")
        return claims
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in extract_claims: {e}")
        print(f"Raw completion: {completion}")
        return []

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def fuzzy_match_claims_to_blocks(claims: List[Dict[str, Any]], text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    matched_claims = []
    
    for claim_index, claim in enumerate(claims):
        print(f"\nMatching claim {claim_index + 1}:")
        print(f"Claim: {claim['statement'][:100]}...")
        
        best_match = None
        best_ratio = 0
        best_substring_ratio = 0
        preprocessed_claim = preprocess_text(claim['statement'])
        
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
            
            matched_claim = claim.copy();
            matched_claim.update({
                'column': best_match['column'],
                'paragraph': best_match['paragraph'],
                'matched_text': best_match['text']
            })
            matched_claims.append(matched_claim)
            print(f"Matched with block:")
            print(f"  Column {best_match['column']}, Paragraph {best_match['paragraph']}")
            print(f"  {best_match['text'][:200]}...")
        else:
            print(f"Failed to match. Best substring ratio: {best_substring_ratio:.2f}, Best fuzzy ratio: {best_ratio:.2f}")
    
    print(f"\nTotal matched claims: {len(matched_claims)}")

    return matched_claims

def get_best_match_line_index(part: str, lines: List[str], start_index=0) -> int:
    best_ratio = 0
    best_index = -1
    for i in range(start_index, len(lines)):
        ratio = SequenceMatcher(None, lines[i].strip(), part.strip()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_index = i
    return best_index

def get_line_range(claim_text: str, block_text: str) -> str:
    block_lines = block_text.split('\n')
    
    # Get the first few and last few words of the claim
    claim_words = claim_text.split()
    first_part = ' '.join(claim_words[:5])
    last_part = ' '.join(claim_words[-5:])
    
    # Find the best matching line for the first part of the claim
    start_line_index = get_best_match_line_index(first_part, block_lines)
    
    # Find the best matching line for the last part of the claim after the starting line
    end_line_index = get_best_match_line_index(last_part, block_lines, start_line_index)
    
    if start_line_index == -1 or end_line_index == -1:
        return "1-" + str(len(block_lines))
    
    start_line = start_line_index +1
    end_line = end_line_index +1
    
    if start_line == end_line:
        return f"{start_line}"
    
    return f"{start_line}-{end_line}"

async def finalize_annotations(client: Anthropic, matched_claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not matched_claims:
        print("No matched claims found.")
        return []

    valid_claims = []
    for claim in matched_claims:
        if 'matched_text' in claim:
            claim['line_range'] = get_line_range(claim['statement'], claim['matched_text'])
            valid_claims.append(claim)
        else:
            print(f"Skipping unmatched claim: {claim['statement'][:100]}...")

    if not valid_claims:
        print("No valid matched claims found after filtering.")
        return []

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Here are claims extracted from a scientific paper, along with their matched locations. For each claim, create the final annotation in the format "FirstAuthorName et al., year, p[Page]/col[Column]/para[Paragraph]/lns[Lines]". Use the page number from the claim, and the column, paragraph, and line range information from the matched location. The year should be extracted from the citation. Make sure page numbers are from the journal itself no page numbers should be 1,2,3... they should all be from the bottom of the pdf. Also make sure you do not change the columns or the paragraphs from the matched claims please. Here are the claims:
{json.dumps(valid_claims, indent=2)}

Please output the results in the following JSON format:
{{
  "extractedClaims": [
    {{
      "statement": "Exact claim text",
      "annotation": "FirstAuthorName et al., year, p[Page]/col[Column]/para[Paragraph]/lns[Lines]",
      "citation": "FirstAuthor et al. Journal Name Volume(Issue):PageRange"
    }},
    // ... more claim objects
  ]
}}

Ensure that the output is valid JSON. Do not include any text before or after the JSON object.
"""
                }
            ]
        }
    ]

    completion = get_completion(client, messages)
    try:
        parsed_json = json.loads(completion)
        if 'extractedClaims' not in parsed_json:
            print(f"Warning: 'extractedClaims' not found in parsed JSON. Raw response: {completion}")
            return []
        return parsed_json['extractedClaims']
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in finalize_annotations: {e}")
        print(f"Raw completion: {completion}")
        
        try:
            json_start = completion.index('{')
            json_end = completion.rindex('}') + 1
            json_str = completion[json_start:json_end]
            parsed_json = json.loads(json_str)
            if 'extractedClaims' in parsed_json:
                return parsed_json['extractedClaims']
        except:
            pass
        
        return []

async def process_pdf_chunk(pdf_path: str, client: Anthropic, start_page: int, end_page: int) -> Dict[str, Any]:
    text_blocks = await extract_text_blocks(pdf_path, client, start_page, end_page)
    full_text, images = await extract_full_text_and_images(pdf_path, start_page, end_page)
    claims = await extract_claims(client, full_text, images)
    matched_claims = fuzzy_match_claims_to_blocks(claims, text_blocks)
    final_claims = await finalize_annotations(client, matched_claims)
    return {"extractedClaims": final_claims, "text_blocks": text_blocks}

async def process_pdf_task(pdf_path: str, task_id: str):
    client = Anthropic(api_key=api_key)

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        chunk_size = 3

        chunk_tasks = []
        for i in range(0, total_pages, chunk_size):
            chunk_tasks.append(process_pdf_chunk(pdf_path, client, i, min(i + chunk_size, total_pages)))

        results = await asyncio.gather(*chunk_tasks)

        combined_claims = []
        combined_text_blocks = []
        for result in results:
            combined_claims.extend(result["extractedClaims"])
            combined_text_blocks.extend(result["text_blocks"])

        # Validate and correct the final claims
        validated_claims = await validate_and_correct_claims(client, combined_claims)

        output_json = {
            "extractedClaims": validated_claims,
            "text_blocks": combined_text_blocks
        }
        
        task_results[task_id] = {"state": "SUCCESS", "result": output_json}
    except Exception as e:
        print(f"Error in process_pdf_task: {str(e)}")
        task_results[task_id] = {"state": "FAILURE", "error": str(e)}
    finally:
        try:
            os.remove(pdf_path)
        except Exception as e:
            print(f"Error cleaning up files: {str(e)}")
class TaskResponse(BaseModel):
    task_id: str

class TaskStatus(BaseModel):
    state: str
    status: str
    result: Dict[str, Any] = None
    error: str = None

@app.post("/process_pdf", response_model=TaskResponse)
async def process_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        return JSONResponse(status_code=400, content={"error": "Invalid file type. Please upload a PDF."})
    
    filename = f"{uuid.uuid4()}.pdf"
    pdf_path = os.path.join(UPLOAD_FOLDER, filename)
    
    with open(pdf_path, "wb") as buffer:
        buffer.write(await file.read())
    
    task_id = str(uuid.uuid4())
    task_results[task_id] = {"state": "PENDING", "status": "Task is pending..."}
    
    background_tasks.add_task(process_pdf_task, pdf_path, task_id)
    
    return {"task_id": task_id}

@app.get("/task_status/{task_id}", response_model=TaskStatus)
async def task_status(task_id: str):
    if task_id not in task_results:
        return TaskStatus(state="UNKNOWN", status="Unknown task ID")
    
    result = task_results[task_id]
    if result["state"] == "PENDING":
        return TaskStatus(state="PENDING", status="Task is still processing...")
    elif result["state"] == "SUCCESS":
        return TaskStatus(state="SUCCESS", status="Task completed", result=result["result"])
    elif result["state"] == "FAILURE":
        return TaskStatus(state="FAILURE", status="Task failed", error=result["error"])
    else:
        return TaskStatus(state="UNKNOWN", status="Unknown task state")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
