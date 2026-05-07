# rag-search-engine

This is a guided project for Boot.dev, covering everything from pre-processing, TF-IDF, semantic and keyword searching, and incorporating LLM's to enhance querries.  This project requires a Gemini API key, but works on free tier. I would prefer to use locally hosted on Ollama, but for the testing in the course, it was a requirement. 

# Technologies used:
- uv: project management, taking care of venv, dependicies, etc.
- google-genai: for LLM access, currently using "gemma-4-26b-a4b-it", free-tier
- numpy: used in vecorization of tokens
- nltk: used in tokenization and stemming of documents
- sentence-transformers: used to create embeddings from documents
- python-dotenv: used to load .env values, specifically the GEMINI_API_KEY
