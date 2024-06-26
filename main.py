import unicodedata
import fitz  # PyMuPDF
from PIL import Image
import io
import os
from anthropic import Anthropic
import base64
import json
import uuid
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories to save uploaded PDFs and extracted images
UPLOAD_FOLDER = 'uploads'
EXTRACTED_IMAGES_FOLDER = 'extracted_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXTRACTED_IMAGES_FOLDER, exist_ok=True)

# Store task results
task_results = {}

def clean_text(text):
    replacements = {
        '\ufb01': 'fi',
        '\ufb02': 'fl',
        '\u2013': '-',
        '\u2014': '-',
        '\u2018': "'",
        '\u2019': "'",
        '\u201c': '"',
        '\u201d': '"',
        '\u2022': '•',
        '\u2026': '...',
        '\u00b6': '¶',  # Pilcrow sign
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')

def extract_text_and_images(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = []
        images = []

        for page_num, page in enumerate(doc):
            try:
                page_text = page.get_text()
                cleaned_text = clean_text(page_text)
                text.append(f"Content of page {page_num + 1}:\n{cleaned_text}\n")
            except Exception as e:
                print(f"Error extracting text from page {page_num + 1}: {str(e)}")

            try:
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        img = Image.open(io.BytesIO(image_bytes))
                        img_path = f"{EXTRACTED_IMAGES_FOLDER}/image_page_{page_num + 1}_{img_index + 1}.png"
                        img.save(img_path)
                        images.append((page_num + 1, img_index + 1, img_path))
                    except Exception as e:
                        print(f"Error saving image {img_index + 1} from page {page_num + 1}: {str(e)}")
            except Exception as e:
                print(f"Error processing images on page {page_num + 1}: {str(e)}")

        return "".join(text), images

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return "", []

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_completion(client, messages):
    return client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0,
        messages=messages
    ).content[0].text

def process_annotation(annotation):
    parts = annotation.split(', ')
    if len(parts) >= 2:
        location = parts[-1]
        return ', '.join(parts[:-1] + [location])
    return annotation

api_key = os.getenv('ANTHROPIC_API_KEY')

async def process_pdf_task(pdf_path: str, task_id: str):
    try:
        client = Anthropic(api_key=api_key)
        text, images = extract_text_and_images(pdf_path)
        
        if not text and not images:
            raise Exception("Failed to extract any content from the PDF")
        
        content = [
            {
                "type": "text",
                "text": f"Here is the text of the academic paper:\n\n{text}\n\nNow I will provide the images from the paper."
            }
        ]

        for page_num, img_num, img_path in images:
            try:
                encoded_image = encode_image(img_path)
                content.extend([
                    {
                        "type": "image",
                        "image": {
                            "type": "base64",
                            "data": encoded_image
                        }
                    },
                    {
                        "type": "text",
                        "text": f"This is image {img_num} from page {page_num} of the paper."
                    }
                ])
            except Exception as e:
                print(f"Error encoding image {img_num} from page {page_num}: {str(e)}")

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
     Page number (use original document footer numbers)
     Column number
     Paragraph number
     Line number(s)

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

        completion = get_completion(client, messages)
        
        try:
            output_json = json.loads(completion)
            
            for claim in output_json['extractedClaims']:
                claim['annotation'] = process_annotation(claim['annotation'])
            
            task_results[task_id] = {"state": "SUCCESS", "result": output_json}
        except json.JSONDecodeError:
            task_results[task_id] = {"state": "FAILURE", "error": "Invalid JSON output", "raw_output": completion}
        
    except Exception as e:
        task_results[task_id] = {"state": "FAILURE", "error": str(e)}
    
    finally:
        # Clean up files
        try:
            os.remove(pdf_path)
            for _, _, img_path in images:
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
    background_tasks.add_task(process_pdf_task, pdf_path, task_id)
    
    return {"task_id": task_id}

@app.get("/task_status/{task_id}", response_model=TaskStatus)
async def task_status(task_id: str):
    if task_id not in task_results:
        return TaskStatus(state="PENDING", status="Task is pending...")
    
    result = task_results[task_id]
    if result["state"] == "SUCCESS":
        return TaskStatus(state="SUCCESS", status="Task completed", result=result["result"])
    elif result["state"] == "FAILURE":
        return TaskStatus(state="FAILURE", status="Task failed", error=result["error"])
    else:
        return TaskStatus(state="UNKNOWN", status="Unknown task state")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
