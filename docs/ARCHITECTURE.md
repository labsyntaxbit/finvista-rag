# FinVista RAG — Architecture Documentation

## System Overview

The Enterprise Financial Intelligence Assistant is a production-ready RAG application that enables natural language interaction with enterprise financial documents.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FinVista RAG System                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────────────────────────────────┐  │
│  │   Streamlit   │    │           RAG Pipeline                  │  │
│  │   Web UI      │───▶│                                         │  │
│  │              │    │  ┌─────────────┐  ┌─────────────────┐  │  │
│  │ • Upload     │    │  │  Document    │  │  Text Chunker   │  │  │
│  │ • Chat       │    │  │  Loader      │──▶│  (1000/200)     │  │  │
│  │ • Citations  │    │  │  (PDF)       │  └────────┬────────┘  │  │
│  │ • History    │    │  └─────────────┘           │            │  │
│  └──────────────┘    │                              ▼            │  │
│                       │                    ┌─────────────────┐   │  │
│                       │                    │  Embedding       │   │  │
│                       │                    │  Service         │   │  │
│                       │                    │  (MiniLM-L6-v2)  │   │  │
│                       │                    └────────┬────────┘   │  │
│                       │                              │            │  │
│                       │                              ▼            │  │
│                       │                    ┌─────────────────┐   │  │
│                       │                    │  ChromaDB        │   │  │
│                       │                    │  Vector Store    │   │  │
│                       │                    └────────┬────────┘   │  │
│                       │                              │            │  │
│                       │  User Query ──▶ ┌───────────▼────────┐  │  │
│                       │                 │  Retriever          │  │  │
│                       │                 │  (Semantic Search)  │  │  │
│                       │                 └───────────┬────────┘  │  │
│                       │                              │            │  │
│                       │                 ┌───────────▼────────┐  │  │
│                       │                 │  Response Generator │  │  │
│                       │                 │  (OpenAI/Ollama)    │  │  │
│                       │                 └───────────┬────────┘  │  │
│                       │                              │            │  │
│                       │                 ┌───────────▼────────┐  │  │
│                       │                 │  Conversation       │  │  │
│                       │                 │  Memory             │  │  │
│                       │                 └────────────────────┘  │  │
│                       └──────────────────────────────────────────┘  │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                     Deployment Layer                                │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  ┌──────────────┐ │
│  │  Docker   │  │  Kubernetes  │  │  CI/CD    │  │  ConfigMap/  │ │
│  │  Container│  │  (EKS)       │  │  GitHub   │  │  Secrets     │ │
│  └──────────┘  └──────────────┘  └───────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Document Ingestion Pipeline

1. User uploads PDF via Streamlit sidebar
2. `DocumentLoader` extracts text from all pages
3. `TextChunker` preprocesses and splits text into overlapping chunks
4. `EmbeddingService` generates vector embeddings for each chunk
5. `VectorStore` indexes chunks in ChromaDB with metadata

### Query Processing Pipeline

1. User submits a natural language question
2. `Retriever` embeds the query and performs semantic search
3. Top-K relevant chunks are retrieved with relevance scores
4. `ResponseGenerator` constructs a prompt with context + conversation history
5. LLM generates a grounded response
6. Source citations are extracted and displayed
7. `ConversationMemory` stores the exchange for follow-up questions

## Component Details

### Document Loader
- Supports PDF format via pypdf
- Extracts text page-by-page
- Validates non-empty content

### Text Chunker
- RecursiveCharacterTextSplitter with configurable size/overlap
- Default: 1000 chars with 200 char overlap
- Preserves document metadata (filename, chunk index)

### Embedding Service
- Model: all-MiniLM-L6-v2 (384 dimensions)
- Runs locally via sentence-transformers
- Singleton pattern for model loading efficiency

### Vector Store
- ChromaDB with persistent storage
- Cosine similarity search
- Metadata filtering by document filename

### Response Generator
- Supports OpenAI, Ollama, and Mock providers
- System prompt enforces context-grounded responses
- Temperature 0.3 for factual accuracy

### Conversation Memory
- Deque-based with configurable max turns
- Provides history context to LLM for multi-turn dialogue

## Deployment Architecture

### Docker
- Python 3.11-slim base image
- Health check on Streamlit endpoint
- Persistent data volumes for ChromaDB and uploads

### Kubernetes (EKS)
- 2 replicas with HPA (2-5 pods)
- ConfigMap for non-sensitive configuration
- Secrets for API keys
- PVCs for ChromaDB (5Gi) and uploads (10Gi)
- Liveness and readiness probes
- LoadBalancer service on port 80

### CI/CD
- Triggered on push to main/master
- Test → Build → Push → Deploy pipeline
- GitHub Container Registry for images
- EKS deployment via kubectl

## Security Considerations

- API keys stored in Kubernetes Secrets, not ConfigMaps
- No secrets in source code or Docker images
- `.env` excluded from version control
- Container runs with resource limits

## Scalability

- Horizontal Pod Autoscaler based on CPU/memory
- ChromaDB persistent storage survives pod restarts
- Embedding model loaded once per pod (cached)
- Stateless query processing enables horizontal scaling
