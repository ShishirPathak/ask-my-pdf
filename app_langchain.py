"""Ask My PDF

This project demonstrates a simple retrieval-augmented generation flow:
1. read a PDF document,
2. split the text into chunks,
3. find the most relevant chunks using embeddings,
4. send the relevant context to Gemini for a grounded answer.
"""

import os
import time

import faiss
import numpy as np
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
PDF_PATH = "sample1.pdf"
QUESTION = "Who is this doc all about?"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K = 3
MAX_RETRIES = 5
MODEL_NAME = "gemini-3.8-flash"

# ------------------------------------------------------------
# PDF extraction and chunking
# ------------------------------------------------------------
def load_pdf_text(pdf_path: str) -> str:
    """Extract text content from all pages in the PDF."""
    reader = PdfReader(pdf_path)
    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    return "\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split the extracted PDF text into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    chunks = splitter.split_text(text)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

# ------------------------------------------------------------
# Retrieval using embeddings
# ------------------------------------------------------------
def find_relevant_chunks(text: str, question: str, top_k: int = TOP_K):
    """Return the most relevant text chunks for a question."""
    chunks = chunk_text(text)
    if not chunks:
        return []

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True).astype("float32")
    question_embedding = model.encode([question], convert_to_numpy=True).astype("float32")

    faiss.normalize_L2(embeddings)
    faiss.normalize_L2(question_embedding)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print("Vectors stored:", index.ntotal)

    scores, top_indices = index.search(question_embedding, min(top_k, len(chunks)))
    relevant_chunks = [chunks[idx] for idx in top_indices[0]]

    print("\n=== Retrieved chunks ===")
    for rank, (score, idx) in enumerate(zip(scores[0], top_indices[0]), start=1):
        print(f"\nSource {rank}")
        print(f"Score: {score:.4f}")
        print(chunks[idx])

    return relevant_chunks

# ------------------------------------------------------------
# Prompt construction and answer generation
# ------------------------------------------------------------
def build_prompt(context: str, question: str) -> str:
    """Create a grounded prompt for the LLM using only the retrieved context."""
    return f"""
You are answering questions based only on the context provided below.
Answer only from the context. If the answer is not in the context, say you do not know.

Context:
{context}

Question: {question}

Answer:
"""


def generate_answer(prompt: str) -> str:
    """Generate a response from Gemini with retry handling."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Export it before running the script.")

    client = genai.Client(api_key=api_key)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            return response.text
        except Exception as exc:
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
    """Run the PDF Q&A pipeline end-to-end."""
    text = load_pdf_text(PDF_PATH)
    relevant_chunks = find_relevant_chunks(text, QUESTION)

    if not relevant_chunks:
        print("No text was found in the PDF.")
        return

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
