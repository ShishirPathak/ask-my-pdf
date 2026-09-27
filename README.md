# Ask My PDF

A simple Python project that reads a PDF, splits it into chunks, finds the most relevant sections using sentence embeddings, and sends the relevant context to Gemini for a grounded answer.

## Project idea

This project demonstrates a lightweight Retrieval-Augmented Generation (RAG) flow:

1. Load a PDF file
2. Extract text from the document
3. Split the text into overlapping chunks
4. Encode the chunks and the user question with a sentence transformer
5. Use cosine similarity to find the most relevant chunks
6. Pass the context to Gemini for an answer

## Features

- PDF text extraction using `pypdf`
- Text chunking for better retrieval
- Semantic similarity search with `SentenceTransformer`
- LLM-powered answer generation using Google Gemini
- Simple retry logic for API failures

## Project structure

```text
ask-my-pdf/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── sample.pdf
└── .venv/   # local virtual environment (ignored by git)
```

## Prerequisites

- Python 3.10+
- A working Google Gemini API setup
- Internet access for model calls

## Installation

```bash
cd "Users/shishirsmac/Personal/AI Learning/Project/ask-my-pdf"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the app

```bash
source .venv/bin/activate
python app.py
```

## Notes

- The app currently uses `sample.pdf` as the default input file.
- You can replace it with your own PDF file by updating the `PDF_PATH` constant in `app.py`.
- If you have a Gemini API key configured in your environment, the app can call the model directly.

## License

This project is for learning and demonstration purposes.
