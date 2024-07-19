import os
import time
from typing import List, Optional, Tuple
import uuid
import re

from fastapi import UploadFile, HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

from app.utils.enums import FileType
from fuzzywuzzy import fuzz


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

def normalize_score(score: float, max_possible_score: float) -> float:
    return score / max_possible_score


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



def find_best_match_using_tfid(subtext: str, text_blocks: List[str], threshold: float = 10.0) -> Optional[Tuple[str, float]]:
    if not text_blocks:
        raise ValueError("text_blocks list is empty")

    all_texts = [text for text in text_blocks]
    
    # Add the normalized subtext to the list of texts
    all_texts.append(subtext)
    
    # Compute TF-IDF vectors for all texts
    vectorizer = TfidfVectorizer().fit_transform(all_texts)
    vectors = vectorizer.toarray()
    
    if vectors.shape[0] < 2:
        raise ValueError("Not enough text data to compute similarities")
    
    # Calculate cosine similarity between the subtext and all document texts
    similarities = cosine_similarity([vectors[-1]], vectors[:-1])[0]
    
    # Convert similarity scores to the range of 0 to 100
    similarities = similarities * 100
    
    # Find the most similar text
    most_similar_index = similarities.argmax()
    most_similar_text = text_blocks[most_similar_index]  # Use original text block for return
    
    similarity_score = similarities[most_similar_index]

    if similarity_score > threshold:
        return most_similar_text, similarity_score
    else:
        return None


def find_best_match_using_sequence_matcher(subtext: str, text_blocks: List[str], threshold: float = 10.0) -> Optional[Tuple[str, float]]:
    if not text_blocks:
        raise ValueError("text_blocks list is empty")

    best_match = None
    highest_score = 0

    for text in text_blocks:
        # Calculate the similarity score using SequenceMatcher
        similarity = SequenceMatcher(None, subtext, text).ratio()
        
        # Convert the similarity score to the range of 0 to 100
        similarity_score = similarity * 100
        
        # Check if this is the highest score found so far
        if similarity_score > highest_score:
            highest_score = similarity_score
            best_match = text

    if highest_score > threshold:
        return best_match, highest_score
    else:
        return None



def combined_best_match(subtext: str, text_blocks: List[str], threshold: float = 5.0) -> Optional[Tuple[int, float]]:
    """
    The combined_best_match function calculates the best match between a given subtext and a list of text blocks
    using a combination of three different similarity measures: TF-IDF similarity, SequenceMatcher similarity,
    and fuzzy word similarity for the first and last 5 words.
    """
    if not text_blocks:
        raise ValueError("text_blocks list is empty")

    all_texts = [text for text in text_blocks]
    all_texts.append(subtext)

    vectorizer = TfidfVectorizer().fit_transform(all_texts)
    vectors = vectorizer.toarray()

    if vectors.shape[0] < 2:
        raise ValueError("Not enough text data to compute similarities")

    tfidf_similarities = cosine_similarity([vectors[-1]], vectors[:-1])[0]

    best_match_index = None
    highest_combined_score = 0

    for index, text in enumerate(text_blocks):
        sequence_similarity = SequenceMatcher(None, subtext, text).ratio()
        tfidf_similarity = normalize_score(tfidf_similarities[index], 1)
        word_similarity = normalize_score(fuzzy_word_similarity(subtext, text), 100)

        # Combine the scores
        combined_score = (0.30 * tfidf_similarity) + (0.20 * sequence_similarity) + (0.50 * word_similarity) 
        combined_score = combined_score / (0.30 + 0.20 + 0.50)  # Normalize combined score
        combined_score = combined_score * 100  # Scale to 1-100

        if combined_score > highest_combined_score:
            highest_combined_score = combined_score
            best_match_index = index

    if highest_combined_score > threshold:
        return best_match_index, highest_combined_score
    else:
        return None


def fuzzy_word_similarity(text1: str, text2: str) -> float:
    words1 = text1.split()
    words2 = text2.split()

    if len(words1) < 5 or len(words2) < 5:
        return 0

    def find_subsequence_similarity(subsequence: List[str], sequence: List[str]) -> float:
        subsequence_str = " ".join(subsequence)
        # sequence_str = " ".join(sequence)
        max_similarity = 0

        # Slide over the sequence to find the best match
        for i in range(len(sequence) - len(subsequence) + 1):
            window = sequence[i:i + len(subsequence)]
            window_str = " ".join(window)
            similarity = fuzz.ratio(subsequence_str, window_str)
            max_similarity = max(max_similarity, similarity)

        return max_similarity

    first_5_similarity = find_subsequence_similarity(words1[:5], words2)
    last_5_similarity = find_subsequence_similarity(words1[-5:], words2)

    return (first_5_similarity + last_5_similarity) / 2
