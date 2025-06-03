# Vision-RAG System

A multi-modal RAG system combining Cohere's embeddings with Gemini's vision capabilities.

## Features
- Image search using Cohere Embed v4
- Question answering with Gemini Flash
- Local image upload and indexing
- Performance metrics tracking

## Setup
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with your API keys (see `.env.example`)

## Running
`streamlit run vision_rag_app.py`