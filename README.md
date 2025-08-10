# Lumina Memory Module

A Flask-based memory management system that uses ChromaDB for vector storage and advanced sentiment analysis for emotional context.

## Features

- Vector-based memory storage using ChromaDB
- Sentiment analysis using both VADER and spaCy
- Rich metadata extraction including:
  - Emotional weight calculation
  - Named entity recognition
  - Key phrase extraction
  - Sentiment categorization
- Memory deduplication with similarity detection
- RESTful API endpoints

## Requirements

- Python 3.8+
- ChromaDB
- spaCy (with en_core_web_sm model)
- VADER Sentiment
- Flask
- llama-cpp-python

## Setup

1. Clone the repository
2. Create a virtual environment:
   \\\ash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   \\\
3. Install dependencies:
   \\\ash
   pip install -r requirements.txt
   \\\
4. Download the Mistral-7B model and place it in the models directory
5. Configure the paths in app_clean.py:
   - Set MODEL_PATH_LOCAL to your local GGUF model path
   - Set the memory_data path for ChromaDB

## API Endpoints

- GET \/status\ - Check server status
- POST \/add_memory\ - Add a new memory
- GET \/get_all_memories\ - Retrieve all stored memories

## Usage

Run the server:
\\\ash
python app_clean.py
\\\

The server will start on http://localhost:5000
