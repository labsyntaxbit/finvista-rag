# FinVista Capital — Enterprise Financial Intelligence Assistant

An AI-powered **Retrieval-Augmented Generation (RAG)** application that enables FinVista Capital employees to interact with enterprise financial documents through natural language conversations.

## Features

- **Document Upload** — Upload PDF financial reports, annual reports, compliance manuals
- **Automatic Processing** — PDF parsing, text preprocessing, and intelligent chunking
- **Semantic Search** — Vector-based similarity search across all indexed documents
- **RAG Pipeline** — Context-aware AI responses grounded in enterprise documents
- **Source Citations** — Every response includes references to source documents
- **Multi-turn Conversations** — Maintains conversation history for follow-up questions
- **Production Ready** — Docker containerization, Kubernetes deployment, CI/CD pipeline

## Architecture

```
User → Streamlit UI → RAG Pipeline → Vector DB (ChromaDB)
                          ├── Document Loader (PDF)
                          ├── Text Chunker
                          ├── Embedding Service (sentence-transformers)
                          ├── Retriever (Semantic Search)
                          ├── LLM Generator (OpenAI/Ollama/Mock)
                          └── Conversation Memory
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| UI | Streamlit |
| PDF Parsing | pypdf |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| LLM | OpenAI GPT / Ollama / Mock (demo) |
| Containerization | Docker |
| Orchestration | Kubernetes (EKS) |
| CI/CD | GitHub Actions |

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Local Development

```bash
# Clone and navigate
cd finvista-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — set LLM_PROVIDER=openai and OPENAI_API_KEY for production

# Run the application
streamlit run app/main.py
```

Open **http://localhost:8501** in your browser.

### Docker

```bash
docker build -t finvista-rag .
docker run -p 8501:8501 -e LLM_PROVIDER=mock finvista-rag
```

### Kubernetes (EKS)

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

## Project Structure

```
finvista-rag/
├── app/
│   ├── main.py                 # Streamlit application
│   ├── rag/
│   │   ├── document_loader.py  # PDF parsing
│   │   ├── chunker.py          # Text chunking
│   │   ├── embeddings.py       # Embedding generation
│   │   ├── vector_store.py     # ChromaDB integration
│   │   ├── retriever.py        # Semantic search
│   │   ├── generator.py        # LLM response generation
│   │   └── memory.py           # Conversation history
│   └── utils/
│       ├── config.py           # Configuration
│       └── logger.py           # Logging
├── k8s/                        # Kubernetes manifests
├── .github/workflows/          # CI/CD pipeline
├── data/                       # Upload & vector store data
├── docs/                       # Documentation
├── Dockerfile
├── requirements.txt
└── README.md
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `mock` | LLM backend: `openai`, `ollama`, or `mock` |
| `OPENAI_API_KEY` | — | OpenAI API key (required for `openai` provider) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CHUNK_SIZE` | `1000` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Number of chunks retrieved per query |

## Usage

1. **Upload Documents** — Use the sidebar to upload PDF financial documents
2. **Process** — Click "Process Documents" to index them
3. **Ask Questions** — Type natural language questions in the chat
4. **Review Citations** — Expand "Source Citations" to see document references
5. **Follow-up** — Ask follow-up questions; conversation history is maintained

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) automates:

1. **Test** — Verify Python imports and dependencies
2. **Build** — Build Docker image and push to GitHub Container Registry
3. **Deploy** — Deploy to EKS cluster on main branch push

Required secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `EKS_CLUSTER_NAME`

## License

Capstone project — FinVista Capital Enterprise GenAI Program.
