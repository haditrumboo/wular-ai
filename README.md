# Wular AI — Kashmir Travel RAG Assistant

Wular AI is a Retrieval-Augmented Generation (RAG) based AI travel assistant for Kashmir. It uses vector search and Large Language Models (LLMs) to provide accurate, context-aware travel recommendations based only on uploaded travel data.

---

## Features

- AI-powered Kashmir travel assistant
- Retrieval-Augmented Generation (RAG)
- Conversational memory support
- Vector similarity search
- Context-aware responses
- Source-based answering
- Fast inference using Groq
- HuggingFace embeddings
- Persistent vector database with ChromaDB

---

## Tech Stack

### Backend
- Python
- LangChain

### LLM
- Groq
- Llama 3.1 8B Instant

### Embeddings
- HuggingFace Embeddings
- BAAI/bge-small-en-v1.5

### Vector Database
- ChromaDB

---

## Project Structure

```bash
wular-ai/
│
├── app/
│   ├── rag.py
│   ├── ingest.py
│   └── prompts.py
│
├── chroma_db/
├── data/
├── .env
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone Repository

```bash
git clone <your-repository-url>
cd wular-ai
```

### 2. Create Virtual Environment

#### Linux / Mac

```bash
python -m venv venv
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## Run the Application

```bash
python rag.py
```

---

## How Wular AI Works

1. Travel documents are loaded into the system
2. Documents are split into chunks
3. Embeddings are generated
4. Embeddings are stored in ChromaDB
5. User query is converted into embeddings
6. Relevant context is retrieved
7. LLM generates the final response using retrieved context

---

## Example Conversation

```text
You: Best places to visit in Gulmarg?

AI: Gulmarg is known for skiing, gondola rides, and scenic mountain views.
```

---

## Retrieval Pipeline

```text
User Query
    ↓
Embedding Model
    ↓
Vector Search (ChromaDB)
    ↓
Relevant Context Retrieval
    ↓
LLM Response Generation
```

---

## Current Capabilities

- Conversational travel assistant
- Context-only answering
- Memory-aware chat
- Semantic search
- Persistent vector storage

---

## Future Improvements

- Web interface
- Voice assistant support
- Streaming responses
- Multi-user chat
- Better reranking
- Hybrid search
- Hotel recommendation system
- Real-time weather integration

---

## Security

- API keys stored using `.env`
- `.env` added to `.gitignore`
- No hardcoded secrets

---

## License

MIT License

---

## Author

Made with ❤️ by Hadi
