# RAG System with Admin & User Separation

A Retrieval-Augmented Generation (RAG) system with separate admin and user interfaces.

## Architecture

This system separates document management from user queries:

- **Admin Interface** (`rag-admin.py`): Upload, process, and index PDF documents
- **User Interface** (`rag-user.py`): Query indexed documents through a chat interface
- **Shared Storage** (`./vector_stores/`): Persisted vector databases accessible by both

## Files

| File | Purpose |
|------|---------|
| `rag-admin.py` | Admin interface for document ingestion and indexing |
| `rag-user.py` | User interface for querying indexed documents |
| `pdf-rag.py` | Original CLI version (preserved) |
| `pdf-rag-streamlit.py` | Single-user web version (preserved) |

## Requirements

- Python 3.11
- Ollama

## Quick Start

### 0. Create Virtual Environment

```bash
cd rag_system
python -m venv .venv
source .venv/bin/activate
```

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Admin Interface

```bash
streamlit run rag-admin.py
```

- Upload PDF documents
- Adjust chunk size/overlap settings if needed
- Documents are automatically indexed and saved to `./vector_stores/`

### 3. Start User Interface (in a separate terminal)

```bash
streamlit run rag-user.py
```

- Select an indexed document from the sidebar
- Ask questions in the chat interface
- Document content is retrieved and used for answers

## Workflow

```
┌─────────────────┐         ┌──────────────────┐
│   Admin Panel   │         │    User Panel    │
│  (rag-admin.py) │         │  (rag-user.py)   │
└────────┬────────┘         └────────┬─────────┘
         │                             │
         │ Upload & Index              │ Select & Query
         ▼                             ▼
┌─────────────────────────────────────────────┐
│            Vector Store Storage             │
│            ./vector_stores/                 │
└─────────────────────────────────────────────┘
```

## Admin Interface Features

- 📤 Upload PDF documents
- ⚙️ Configure chunk size, overlap, and embedding model
- 📊 View processing results (pages, chunks)
- 📚 Manage indexed documents (list, delete)
- 🔗 Copy collection names for sharing

## User Interface Features

- 📚 Select from available indexed documents
- 💬 Interactive chat interface
- 🔍 AI-powered document retrieval
- 📊 View document statistics (chunks indexed)
- ⚙️ Adjustable query temperature
- 📝 Persistent chat history per session

## Configuration

### Admin Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Chunk Size | 1200 | Characters per chunk |
| Chunk Overlap | 300 | Overlapping characters |
| Embedding Model | nomic-embed-text | Model for embeddings |

### User Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Temperature | 0.1 | Response creativity (0=focused, 1=creative) |

## Directory Structure

```
rag_system/
├── rag-admin.py          # Admin interface
├── rag-user.py           # User interface
├── pdf-rag.py            # Original CLI version
├── pdf-rag-streamlit.py  # Single-user web version
├── requirements.txt      # Dependencies
├── vector_stores/        # Indexed documents (created automatically)
└── data/                 # Source documents (optional)
```

## Running Both Interfaces

You can run both interfaces simultaneously on different ports:

```bash
# Terminal 1 - Admin
streamlit run rag-admin.py --server.port 8501

# Terminal 2 - User  
streamlit run rag-user.py --server.port 8502
```

Then access:
- Admin: http://localhost:8501
- User: http://localhost:8502

## Requirements

- Python 3.8+
- Ollama running locally with required models
- See `requirements.txt` for Python dependencies

## Docker Deployment

The complete system can be deployed using Docker Compose, which includes Ollama with pre-configured models.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- (Optional) [NVIDIA Docker Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support

### Quick Start with Docker

1. **Clone and navigate to the project:**

```bash
cd rag_system
```

2. **Start all services:**

```bash
docker-compose up -d
```

This will start three services:
- **ollama** (port 11434): LLM and embedding model server
- **rag-admin** (port 8501): Admin interface for document processing
- **rag-user** (port 8502): User interface for querying documents

3. **Wait for models to download:**

On first run, Ollama will automatically download the required models:
- `nomic-embed-text` (embedding model)
- `llama3.2:latest` (language model)

This may take several minutes depending on your connection speed.

4. **Access the interfaces:**

- **Admin Panel**: http://localhost:8501
- **User Panel**: http://localhost:8502
- **Ollama API**: http://localhost:11434

### Docker Commands

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# View specific service logs
docker compose logs -f ollama
docker compose logs -f rag-admin
docker compose logs -f rag-user

# Stop all services
docker compose down

# Stop and remove volumes (deletes all indexed documents!)
docker compose down -v

# Rebuild after code changes
docker compose up -d --build

# Check service status
docker compose ps
```

### Docker Configuration

#### GPU Support (Optional)

By default, the `docker-compose.yml` includes GPU support for Ollama. For CPU-only deployment, remove or comment out the `deploy` section in the `ollama` service:

```yaml
# Remove this section for CPU-only
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `STREAMLIT_SERVER_PORT` | `8501` / `8502` | Streamlit server port |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | Streamlit bind address |

#### Volumes

| Volume | Purpose |
|--------|---------|
| `ollama_data` | Persisted Ollama models |
| `vector_stores` | Persisted document indexes |
| `./data` | PDF files for processing (mounted read-only to admin) |

**Adding PDFs for processing:**
Place PDF files in the `./data/` directory before starting the containers, or upload directly through the admin web interface.

### Docker Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network                           │
│                  (rag-network)                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐   │
│  │   rag-admin  │◄────►│    ollama    │◄────►│ rag-user │   │
│  │   :8501      │      │   :11434     │      │  :8502   │   │
│  └──────────────┘      └──────────────┘      └──────────┘   │
│        │                                              │     │
│        ▼                                              ▼     │
│  ┌──────────────┐                              ┌────────┐   │
│  │ vector_stores│                              │  read  │   │
│  │   (volume)   │                              │  only  │   │
│  └──────────────┘                              └────────┘   │
│                                                             │
│  ┌──────────────┐                                           │
│  │ ollama_data  │                                           │
│  │   (volume)   │                                           │
│  └──────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Troubleshooting Docker

**Issue: Services fail to start**
```bash
# Check logs for errors
docker-compose logs

# Verify Ollama is healthy
curl http://localhost:11434/api/tags
```

**Issue: Models not found**
```bash
# Manually pull models
docker-compose exec ollama ollama pull nomic-embed-text
docker-compose exec ollama ollama pull llama3.2:latest
```

**Issue: Permission errors with volumes**
```bash
# Fix permissions on Linux
sudo chown -R $USER:$USER ./vector_stores
```

**Issue: GPU not detected**
- Verify NVIDIA Docker Toolkit is installed
- Check `nvidia-smi` works on host
- Remove GPU section from docker-compose.yml for CPU-only mode

## Notes

- Vector stores are persisted in `./vector_stores/` directory (or Docker volume)
- Each document gets a unique collection name
- The embedding model used for indexing must match when querying
- Users can only query documents that have been indexed by an admin
- Docker volumes persist data across container restarts
