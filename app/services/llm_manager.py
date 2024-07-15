import json
from typing import List, Any, Dict

from anthropic import Anthropic
import requests
from openai import AzureOpenAI
from app.config import settings
from app.pydantic_schemas.claims_extraction_task import ExtractedImage
from app.utils.utils import retry_operation
from tenacity import AsyncRetrying,retry, stop_after_attempt, wait_fixed


class LLMManager:
    def __init__(self):
        self.openai_api_key = settings.OPENAI_API_KEY
        self.client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )

    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
    def convert_page_as_image_to_formatted_json(base64_image: str) -> str:
        azure_endpoint = (
            f"https://{settings.AZURE_RESOURCE_NAME}.openai.azure.com/openai/"
            f"deployments/{settings.AZURE_OPENAI_MODEL}/chat/"
            f"completions?api-version={settings.AZURE_OPENAI_API_VERSION}"
        )
        headers = {
            "Content-Type": "application/json",
            "api-key": settings.AZURE_OPENAI_API_KEY,
        }

        json_format_sample = {"page_markdown": "string"}

        json_format_sample = json.dumps(json_format_sample)

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a very advanced OCR for extracting text in clinical document page images. "
                            "Your task is to convert the contents of these images into markdown "
                            "format. Please adhere to the following guidelines: \n\n"
                            "1. Scan the image thoroughly, starting from the top-left corner and "
                            "progressing in a logical reading order.\n"
                            "2. Capture all visible elements on the page, including text, tables, "
                            "figures, diagrams, and any other relevant information, "
                            "please add `\\n` for new lines .\n"
                            "3. Use the following JSON structure to response the page as markdown structure:\n"
                            f"{json_format_sample} \n\n"
                            "4. Be as precise and accurate as possible in your extraction and "
                            "interpretation of the document contents.\n",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"{base64_image}",
                            },
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        try:
            response = requests.post(
                azure_endpoint, headers=headers, json=payload, timeout=150
            )
            response_json = response.json()
            if "error" in response_json:
                raise Exception(response_json["error"]["message"])
            print("Converted pdf file page to formatted JSON.")
            return response_json["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Exception in convert_page_as_image_to_formatted_json: {e}")
            raise Exception(f"Failed to convert image to text: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
    def match_claims_to_document(self, text: str, base64_image: str):
        print("Match claims in page and it's metadata.")
        json_output_format = {
            "response": {
                "isTextFound": "bool",
                "TextFounded": "string",
                "MatchedTextColumnNumber": "int",
                "MatchedTextParagraphNumber": "int",
                "MatchedTextStartingLineNumber": "int",
                "MatchedTextEndingLineNumber": "int",
            }
        }

        messages = [
            {
                "role": "user",
                "content": "You are a highly capable VLM OCR model, your main task is to search for a text that I'll "
                "send you and search for it it in an image. If you find the text in the image, "
                f"please respond with the following JSON format {json_output_format}. If you "
                "I want you to extract also the column number and paragraph of the text and the "
                "starting line number and ending line number of the text in the paragraph."
                "Please follow these questions to reach the results needed:\n\n"
                "1. Could you locate the text? \n"
                "2. How many columns are there on this page layout? \n"
                "3. From which section did you extract this text? \n"
                "4. How many columns does this section contain? \n"
                "5. How many paragraphs are in this section? \n"
                "6. What is the starting line number of the extracted text in the section, and what is the "
                "ending line number? \n",
            },
            {
                "role": "user",
                "content": f"Here is the text of the claim:\n\n"
                f"{text}\n\nHere is the image:\n\n"
                f"{base64_image}",
            },
        ]
        # Stage 1: Extract the claims from the text
        response = self.client.chat.completions.create(
            model=settings.AZURE_OPENAI_MODEL,
            temperature=0,
            messages=messages,
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content).get("response", {})

    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
    def extract_page_number_from_image(base64_image: str):
        def extract_page_number_from_image(base64_image: str):
            azure_endpoint = (
                f"https://{settings.AZURE_RESOURCE_NAME}.openai.azure.com/openai/"
                f"deployments/{settings.AZURE_OPENAI_MODEL}/chat/"
                f"completions?api-version={settings.AZURE_OPENAI_API_VERSION}"
            )
            headers = {
                "Content-Type": "application/json",
                "api-key": settings.AZURE_OPENAI_API_KEY,
            }

            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are a vision model, your main task is to extract page number "
                                "from pdf page image. "
                                "If you couldn't find the page number in pdf page image response "
                                "with this format {location:string, lineExtracted: string, page_number: number}. we use 0 as our default value. "
                                "I want all your responses to be in the same JSON format, if you "
                                "could locate text please response in this format {location:string, lineExtracted: string, page_number: number}."
                                "The page numbering is usually located at the bottom (Footer) of the page or on the top (Header) section.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 300,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
            timeout_value = 90  # Timeout in seconds
            response = requests.post(
                azure_endpoint, headers=headers, json=payload, timeout=timeout_value
            )
            response_json = response.json()

            if "error" in response_json:
                raise Exception(response_json["error"]["message"])

            if "choices" not in response_json:
                return {"page_number": 1}

            model_response_as_json = response_json["choices"][0]["message"]["content"]
            print(model_response_as_json)
            print("Extracted Correct page number out of text successfully.")
            # Ensure that the parsed JSON is a list of dictionaries
            statements_data = json.loads(model_response_as_json).get("page_number", 1)
            page_number: int = statements_data

            return page_number

        try:
            return retry_operation(
                extract_page_number_from_image,
                retries=3,
                delay=10,
                base64_image=base64_image,
            )
        except Exception as e:
            print("Failed to convert image to text after retries: ", e)
            return {"isImageContainText": False, "text": ""}

    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
    def classify_text_with_claude(client: Anthropic, text: str) -> bool:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Imagine you are an annotator trying to identify the paragraph number for each claim in this scientific paper. "
                        "In order to do this accurately, please classify the following text block as either part of the core scientific article "
                        "text (true) or not (false) so that you can annotate correctly . All titles, footnotes, author names, miscellaneous other "
                        "information about the paper should be false, only text that is a part of the paper itself and its content should be true "
                        "this include purpose, methods, results, etc..Respond with only 'true' or 'false'.\n\n"
                        "Text to classify:\n"
                        f"{text}\n"
                        "Is this part of the scientific article text?",
                    }
                ],
            }
        ]

        response = (
            client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1,
                temperature=0,
                messages=messages,
            )
            .content[0]
            .text.strip()
            .lower()
        )

        return response == "true"

    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
    def extract_claims_with_claude(client: Anthropic, full_text: str, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            content = [
                {
                    "type": "text",
                    "text": f"Here is the text of the academic paper:\n\n{full_text}\n\nNow I will provide the images from the paper, if any were successfully extracted.",
                }
            ]

            accepted_image_formats = {"jpeg", "png", "gif", "webp"}

            for image in images:
                if not isinstance(image, dict):
                    print(f"Skipping image because it is not a dict:")
                    continue

                if image.get("image_format", "").lower() not in accepted_image_formats:
                    print(f"Skipping image {image.get('image_index')} from page {image.get('page_number')} due to unsupported format: {image.get('image_format')}")
                    continue

                try:
                    content.append({"type": "text", "text": f"Image {image.get('image_index')} from page {image.get('page_number')}"})
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{image.get('image_format')}",
                            "data": image.get("base64_data"),
                        },
                    })
                except Exception as e:
                    print(f"Error processing image {image.get('image_index')} from page {image.get('page_number')}: {str(e)}")

            content.append(
                {
                    "type": "text",
                    "text":  "You are a highly capable AI assistant tasked with extracting and organizing information from a scientific paper about a drug to then be used on the drug's website. Follow these instructions carefully:\n"
                            "1. CLAIM EXTRACTION:\n" 
                            "   - Identify all claims related to:\n"
                            "        a. Study design\n"
                            "        b. Patient outcomes and primary and secondary endpoints\n"
                            "        c. Efficacy of drug in treating a specific disease compared to control. Common efficacy metrics include progression free survival (pfs), overall survival (os), objective response rate (ORR), reduction in risk of death, etc.\n"
                            "        d. Adverse events associated with drug\n"
                            "   - Include claims ranging from phrases to full paragraphs or tables\n"
                            "   - Focus on extracting claims that are similar in style and content to the following examples:\n"
                            "2. SOURCE IDENTIFICATION:\n"
                            "   - For each claim, note:\n"
                            "       - Page number (use original document footer numbers)\n"
                            "   - For each claim, determine if it was found in the abstract, introduction, methodology, results, discussion, or conclusion section. Only classify based on these 5 sections. Try to have at least a few claims from each\n"
                            "       - Citation in the format: \"FirstAuthor et al. Journal Name Volume(Issue):PageRange\"\n"
                            "3. JSON OUTPUT STRUCTURE:\n"
                            "   Create a JSON object with the following structure:\n"
                            "   {\n"
                            "     \"extractedClaims\": [\n"
                            "         {\n"
                            "             \"statement\": \"Exact claim text\",\n"
                            "             \"page\": \"Page number as listed in the document\",\n"
                            "             \"citation\": \"FirstAuthor et al. Journal Name Volume(Issue):PageRange\",\n"
                            "             \"Section\": \"introduction\"\n"
                            "         },\n"
                            "         // ... more claim objects\n"
                            "     ]\n"
                            "   }\n"
                            "4. SPECIAL CASES:\n"
                            "   - Multi-page claims: Indicate full range in page field\n"
                            "   - Missing info: Use null for missing fields\n"
                            "5. JSON FORMATTING:\n"
                            "   - Ensure valid JSON format\n"
                            "   - Use double quotes for strings\n"
                            "   - Format JSON for readability with appropriate indentation\n"
                            "6. PROCESSING INSTRUCTIONS:\n"
                            "   - Analyze the entire document before starting output\n"
                            "   - Prioritize accuracy over speed\n"
                            "   - If uncertain about a claim's relevance, include it\n"
                            "   - Do not summarize or paraphrase claims; use exact text\n"
                            "   - Try to combine claims that are near each and about the same topic when possible without reducing quality.\n"
                            "7. SELF-CHECKING:\n"
                            "   - Verify all extracted information meets specified criteria\n"
                            "   - Make sure each claim is relevant to demonstrating the drug's efficacy, adverse events associated with the drug, or study design that would be relevant to a patient or physician interested in the drug. If it is not, then remove the entry from the JSON.\n"
                            "   - Double-check page numbers for accuracy\n"
                            "   - Make sure each claim was classified into its section and the section provided is one of either 'abstract','introduction','methodology','results','discussion','conclusion',' \n"
                            "   - Ensure JSON is well-formed and valid\n"
                            "   - Make sure page numbers are accurate\n"
                            "   - Make sure all citations are consistent\n"
                            "Begin your output with the JSON object as specified in step 3. Do not include any text before or after the JSON output.",
                }
            )

            messages = [{"role": "user", "content": content}]

            try:
                completion = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=4000,
                    temperature=0,
                    messages=messages,
                )
                parsed_json = json.loads(completion.content[0].text)
                print(parsed_json)
                if "extractedClaims" not in parsed_json:
                    print(
                        f"Warning: 'extractedClaims' not found in parsed JSON. Raw response: {completion}"
                    )
                    return []
                claims = parsed_json["extractedClaims"]
                print(claims)

                # Filter out claims not in the desired sections
                desired_sections = {"introduction", "methodology", "results"}
                filtered_claims = [claim for claim in claims if claim.get('Section') in desired_sections]
                
                return filtered_claims
            except Exception as e:
                print(f"Error parsing JSON in extract_claims: {e}")
                return []
        except Exception as e:
            print(f"Error in extract_claims: {e}")
