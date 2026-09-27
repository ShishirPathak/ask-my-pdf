"""Ask My PDF

A lightweight retrieval-augmented generation example that:
1. reads a PDF document,
2. splits it into small chunks,
3. finds the most relevant chunks using embeddings,
4. sends the relevant context to Gemini for a grounded answer.
"""

import time

import numpy as np
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
PDF_PATH = "sample.pdf"
QUESTION = "How does RAG help a company answer questions using its own documents?"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
TOP_K = 3
MAX_RETRIES = 5
MODEL_NAME = "gemini-3.8-flash"

# ------------------------------------------------------------
# PDF extraction and chunking
# ------------------------------------------------------------
def load_pdf_text(pdf_path: str) -> str:
    """Extract text from each page of a PDF file."""
    reader = PdfReader(pdf_path)
    text_parts = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)

    return "\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split long text into overlapping chunks for retrieval."""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

# ------------------------------------------------------------
# Retrieval using embeddings
# ------------------------------------------------------------
def find_relevant_chunks(text: str, question: str, top_k: int = TOP_K):
    """Return the most relevant text chunks for a question."""
    chunks = chunk_text(text)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(chunks)
    question_embedding = model.encode([question])
    similarities = cosine_similarity(question_embedding, embeddings)[0]

    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[idx] for idx in top_indices]

# ------------------------------------------------------------
# Prompt construction and answer generation
# ------------------------------------------------------------
def build_prompt(context: str, question: str) -> str:
    """Create a grounded prompt for the LLM."""
    return f"""
You are answering questions based only on the context provided below.

Context:
{context}

Question: {question}

Answer:
"""


def generate_answer(prompt: str) -> str:
    """Generate a response from Gemini with simple retry logic."""
    client = genai.Client()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            return response.text
        except Exception as exc:  # Broad catch to keep the example readable.
            print(f"Attempt {attempt} failed: {exc}")

            if attempt < MAX_RETRIES:
                wait_seconds = 2 ** (attempt - 1)
                print(f"Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
            else:
                print("Gemini is still unavailable. Try again later.")
                return "Unable to generate an answer right now."

# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------
def main():
    """Run the PDF Q&A pipeline."""
    text = load_pdf_text(PDF_PATH)
    relevant_chunks = find_relevant_chunks(text, QUESTION)
    context = "\n\n".join(relevant_chunks)
    prompt = build_prompt(context, QUESTION)

    print("=== Relevant context ===")
    print(context)
    print("\n=== Prompt sent to Gemini ===")
    print(prompt)

    answer = generate_answer(prompt)
    print("\n=== Final answer ===")
    print(answer)


if __name__ == "__main__":
    main()
