# Ask My PDF

A Python project that reads a PDF, extracts text, creates smart chunks, retrieves the most relevant sections using embeddings and FAISS, and then sends the relevant context to Google Gemini for a grounded answer.

## Overview

This project demonstrates a lightweight Retrieval-Augmented Generation (RAG) pipeline:

1. Read a PDF file
2. Extract text from all pages
3. Split the text into overlapping chunks
4. Convert chunks and the question into embeddings
5. Use FAISS similarity search to find the best matches
6. Send the selected context to Gemini
7. Return a grounded answer based only on the document content

## Features

- PDF extraction using `pypdf`
- Recursive chunking with `langchain-text-splitters`
- Semantic retrieval with `SentenceTransformer` + `FAISS`
- Gemini API integration for answer generation
- Retry handling for transient API errors
- Clean, modular Python structure

## Project structure

```text
ask-my-pdf/
├── app.py
├── app_langchain.py
├── README.md
├── requirements.txt
├── .gitignore
├── sample.pdf
├── sample1.pdf
└── .venv/   # local Python environment
```

## Prerequisites

- Python 3.10+
- Git
- A Google Gemini API key
- Internet access for the Gemini API call

## Setup

```bash
cd "/Users/shishirsmac/Personal/AI Learning/Project/ask-my-pdf"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Required environment variable

Before running the app, export your Gemini API key:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

## Run the app

```bash
source .venv/bin/activate
python app_langchain.py
```

You can also use the simpler version:

```bash
python app.py
```

## Notes

- The default PDF file is `sample.pdf`.
- You can change the input file by editing the `PDF_PATH` constant in `app_langchain.py`.
- The app is designed for learning and experimentation with RAG, embeddings, and LLM-grounded question answering.

## License

This project is for learning and demonstration purposes.
