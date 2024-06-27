import base64
import json
import os
import uuid
from typing import List, Dict, Any, Tuple
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from anthropic import Anthropic
import fitz  # PyMuPDF
from PIL import Image
import concurrent.futures

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories to save uploaded files and extracted images
UPLOAD_FOLDER = 'uploads'
EXTRACTED_IMAGES_FOLDER = 'extracted_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXTRACTED_IMAGES_FOLDER, exist_ok=True)

# Store task results
task_results = {}

# It's better to use an environment variable for the API key
api_key = os.get_env("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

executor = concurrent.futures.ThreadPoolExecutor()

def extract_text_and_images(pdf_path: str) -> Tuple[str, List[Tuple[int, int, str]]]:
    text = []
    images = []
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text.append(page.get_text())
        
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(BytesIO(image_bytes))
                img_filename = f"image_page_{page_num+1}_{img_index+1}.png"
                img_path = os.path.join(EXTRACTED_IMAGES_FOLDER, img_filename)
                image.save(img_path)
                images.append((page_num+1, img_index+1, img_path))
            except Exception as e:
                print(f"Error extracting image {img_index+1} from page {page_num+1}: {str(e)}")
    
    return "\n".join(text), images

def extract_text_only(pdf_path: str) -> str:
    text = []
    pdf_document = fitz.open(pdf_path)
    
    for page in pdf_document:
        try:
            text.append(page.get_text())
        except Exception as e:
            print(f"Error extracting text from page {page.number + 1}: {str(e)}")
    
    return "\n".join(text)

def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_completion(client: Anthropic, messages: List[Dict[str, Any]]) -> str:
    return client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
        temperature=0,
        messages=messages
    ).content[0].text

def process_annotation(annotation: str) -> str:
    parts = annotation.split(', ')
    if len(parts) >= 2:
        location = parts[-1]
        return ', '.join(parts[:-1] + [location])
    return annotation

def escape_json_string(s: str) -> str:
    return s.replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')

def process_pdf_task(pdf_path: str, task_id: str):
    client = Anthropic(api_key=api_key)
    images = []
    content = []

    try:
        # First attempt: Process with images
        text, images = extract_text_and_images(pdf_path)
        text = escape_json_string(text)  # Escape text for JSON

        content = [
            {
                "type": "text",
                "text": f"Here is the text of the academic paper:\n\n{text}\n\nNow I will provide the images from the paper, if any were successfully extracted."
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
    except Exception as e:
        print(f"Error processing PDF with images: {str(e)}. Falling back to text-only processing.")
        # Second attempt: Fall back to text-only processing
        text = extract_text_only(pdf_path)
        text = escape_json_string(text)  # Escape text for JSON

        content = [
            {
                "type": "text",
                "text": f"Here is the text of the academic paper:\n\n{text}\n\nNo images could be extracted from this paper due to processing limitations."
            }
        ]

    content.append({
        "type": "text",
        "text": """You are a highly capable AI assistant tasked with extracting and organizing information from a scientific paper about a drug to then be used on the drug's website. Follow these instructions carefully:

1. CLAIM EXTRACTION:
   - Identify all claims related to: 
     a. Study design 
     b. Patient outcomes and primary and secondary endpoints
     c. Efficacy of drug in treating a specific disease compared to control. Common efficacy metrics include progression free survival (pfs), overall survival (os), objective response rate (ORR), reduction in risk of death, etc.  
     d. Adverse events associated with drug 
   - Include claims ranging from phrases to full paragraphs or tables
   - Focus on extracting claims that are similar in style and content to the following examples:

2. SOURCE IDENTIFICATION:
   - For each claim, note:
- Page number (use original document footer numbers)
- Column number (refer to the left column as column 1 and refer to the right column as column 2)
- Paragraph number (begin count at the start of a column and every double entered paragraph is. Count every new block of text with an indent or after an enter as a new paragraph. Count the first block of text of each column as paragraph. Do not count tables or images as paragraphs. Do not count footnotes or table descriptions as paragraphs.)
- Line number(s) (begin count at the start of a paragraph)

3. JSON OUTPUT STRUCTURE:
   Create a JSON object with the following structure:
   { "extractedClaims": [
       {
         "statement": "Exact claim text",
         "annotation": "FirstAuthorName et al., year, p[Page]/col[Column]/para[Paragraph]/lns[Lines]",
         "citation": "FirstAuthor et al. Journal Name Volume(Issue):PageRange"
       },
       // ... more claim objects
     ]
   }

4. SPECIAL CASES:
   - Multi-page claims: Indicate full range in annotation
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
   - Double-check location identifiers for accuracy
   - Ensure JSON is well-formed and valid
   - Make sure all citations are consistent and all annotations follow the same format

Begin your output with the JSON object as specified in step 3. Do not include any text before or after the JSON output."""
    })

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    try:
        completion = get_completion(client, messages)
        output_json = json.loads(completion)
        
        for claim in output_json['extractedClaims']:
            claim['annotation'] = process_annotation(claim['annotation'])
        
        task_results[task_id] = {"state": "SUCCESS", "result": output_json}
    except Exception as e:
        print(f"Error during completion with images: {str(e)}. Retrying with text-only processing.")
        # If any error occurs, retry with text-only processing
        try:
            text = extract_text_only(pdf_path)
            text = escape_json_string(text)  # Escape text for JSON

            content = [
                {
                    "type": "text",
                    "text": f"Here is the text of the academic paper:\n\n{text}\n\nNo images could be extracted from this paper due to processing limitations."
                },
                {
                    "type": "text",
                    "text": """You are a highly capable AI assistant tasked with extracting and organizing information from a scientific paper about a drug to then be used on the drug's website. Follow these instructions carefully:

1. CLAIM EXTRACTION:
   - Identify all claims related to: 
     a. Study design 
     b. Patient outcomes and primary and secondary endpoints
     c. Efficacy of drug in treating a specific disease compared to control. Common efficacy metrics include progression free survival (pfs), overall survival (os), objective response rate (ORR), reduction in risk of death, etc.  
     d. Adverse events associated with drug 
   - Include claims ranging from phrases to full paragraphs or tables
   - Focus on extracting claims that are similar in style and content to the following examples:

2. SOURCE IDENTIFICATION:
   - For each claim, note:
- Page number (use original document footer numbers)
- Column number (refer to the left column as column 1 and refer to the right column as column 2)
- Paragraph number (begin count at the start of a column and every double entered paragraph is. Count every new block of text with an indent or after an enter as a new paragraph. Count the first block of text of each column as paragraph. Do not count tables or images as paragraphs. Do not count footnotes or table descriptions as paragraphs.)
- Line number(s) (begin count at the start of a paragraph)

3. JSON OUTPUT STRUCTURE:
   Create a JSON object with the following structure:
   { "extractedClaims": [
       {
         "statement": "Exact claim text",
         "annotation": "FirstAuthorName et al., year, p[Page]/col[Column]/para[Paragraph]/lns[Lines]",
         "citation": "FirstAuthor et al. Journal Name Volume(Issue):PageRange"
       },
       // ... more claim objects
     ]
   }

4. SPECIAL CASES:
   - Multi-page claims: Indicate full range in annotation
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
   - Double-check location identifiers for accuracy
   - Ensure JSON is well-formed and valid
   - Make sure all citations are consistent and all annotations follow the same format

Begin your output with the JSON object as specified in step 3. Do not include any text before or after the JSON output."""
                }
            ]

            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]

            try:
                completion = get_completion(client, messages)
                output_json = json.loads(completion)
                
                for claim in output_json['extractedClaims']:
                    claim['annotation'] = process_annotation(claim['annotation'])
                
                task_results[task_id] = {"state": "SUCCESS", "result": output_json}
            except Exception as e:
                task_results[task_id] = {"state": "FAILURE", "error": str(e)}
        except Exception as e:
            task_results[task_id] = {"state": "FAILURE", "error": str(e)}
    finally:
        # Clean up files
        try:
            os.remove(pdf_path)
            for _, _, img_path in images:
                if os.path.exists(img_path):
                    os.remove(img_path)
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
    
    # Use ThreadPoolExecutor to handle the task in the background
    executor.submit(process_pdf_task, pdf_path, task_id)
    
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
