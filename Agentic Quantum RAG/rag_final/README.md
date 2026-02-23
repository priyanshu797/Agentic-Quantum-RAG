# Agentic RAG

A fully agentic retrieval-augmented generation system that runs entirely in the terminal. The agent autonomously decides whether to search documents, query the web, run deep multi-query research, or answer directly — based on the nature of each question.

---

## Architecture

```
You (terminal input)
        |
        v
  Orchestrator Agent   <-- THINK: classifies intent, selects tool, generates sub-queries
        |
   Tool Registry
   |      |      |         |          |
Vector  Web   Deep      Memory    Direct
Search  Search Research  Tool      Answer
   |      |      |
   v      v      v
Context Builder  <-- Merge + deduplicate + token compress
        |
        v
Reasoning Agent  <-- Async Groq LLM: generate grounded answer (bullet points)
        |
        v
Verifier Agent   <-- Async Groq LLM: hallucination check, groundedness score
        |
   Loop-back if failed
        |
        v
Redis Semantic Cache  <-- L1 (in-memory LRU) + L2 (Redis HNSW vector index)
        |
        v
  Output to terminal
```

---

## Features

### Agentic Orchestrator
- Intent classification per query (factual, analytical, research, conversational)
- Dynamic tool selection via LLM decision
- Aware of whether documents are loaded (adjusts routing accordingly)
- Multi-step THINK -> ACT -> OBSERVE -> REFINE loop
- Automatic query rewriting and retry when retrieval quality is low
- Configurable max retries and quality threshold

### Tool System

**Vector Retriever Tool**
- Semantic search via LlamaIndex VectorStoreIndex
- ChromaDB persistent vector database
- Chunk size 512, overlap 50 via LlamaIndex SentenceSplitter
- K_INIT=20 candidates retrieved via ANN
- Threshold filter
- K_RERANK=10 fed into CrossEncoder (cross-encoder/ms-marco-MiniLM-L-6-v2)
- K_FINAL=5 best results passed to context builder
- Cache-first lookup before any retrieval

**Web Search Tool**
- Async HTTP via aiohttp
- Tavily API when TAVILY_API_KEY is set
- Mock fallback when no key is present

**Deep Research Tool**
- Agent generates 3-5 specific sub-queries for the topic
- All queries run in parallel via asyncio.gather
- Results merged and deduplicated by URL
- Returns comprehensive multi-source context

**Memory Tool**
- Session-scoped conversation history
- Injects last N turns as context
- Used when user refers to previous messages

**Direct Answer Tool**
- Used only for pure greetings
- No retrieval, direct LLM response

### Multi-Step Reasoning Loop
- Orchestrator runs up to MAX_RETRIES+1 iterations
- Each iteration: select tool, run tool, score quality
- If quality below MIN_QUALITY threshold: rewrite query and retry
- Best result across all iterations is kept
- Agent knows when no documents are loaded and routes to web/direct accordingly

### Context Builder
- Merges K_FINAL reranked chunks
- Sentence-level deduplication across chunks
- Token budget enforcement (default 1500 tokens)
- Source annotation per chunk

### Async LLM Generation
- ReasoningAgent uses AsyncGroq client
- VerifierAgent uses AsyncGroq client
- Both run async calls cleanly from synchronous terminal context
- Output formatted as bullet points, one fact per line

### Verifier / Critic Agent
- Scores groundedness (0.0-1.0)
- Scores responsiveness (0.0-1.0)
- Detects hallucination risk
- Detects conflicts in retrieved context
- Triggers automatic regeneration with stricter prompt if answer fails
- Verified answers are cached; unverified answers are not

### Redis Semantic Cache
- L1: in-process OrderedDict LRU (256 entries, instant lookup)
- L2: Redis HNSW vector index with cosine similarity
- Exact match on MD5 hash of query
- Semantic match via cosine similarity threshold (default 0.95)
- Configurable TTL (default 24 hours)
- Graceful fallback to L1-only when Redis is not running
- /cache command shows live stats

### Document Support
| Format | Method |
|--------|--------|
| PDF    | pdfplumber |
| DOCX   | python-docx |
| TXT, MD | plain read |
| CSV    | csv reader |
| JSON   | json.load |
| Images | pytesseract OCR |
| Audio  | OpenAI Whisper |
| ZIP    | recursive extraction |

---

## Setup

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Configure environment

```
cp .env.example .env
```

Open .env and fill in:

```
GROQ_API_KEY=your_key_here         # required - get free at console.groq.com
TAVILY_API_KEY=your_key_here       # optional - for real web search
```

Redis is optional. The system works without it using in-memory cache only.

### 3. Run

```
cd rag_final
python main.py
```

### 4. Load documents

When prompted, enter file paths:
```
Enter file path(s) (comma-separated) or press Enter to skip:
Files: C:\docs\report.pdf, C:\docs\notes.txt
```

Or pass at startup:
```
python main.py --files report.pdf notes.txt
python main.py --reset   (wipe DB and start fresh)
```

### Terminal Commands

| Command    | Action |
|------------|--------|
| /files     | Load additional documents |
| /reset     | Wipe vector DB and cache, start fresh |
| /history   | Show conversation history |
| /cache     | Show cache statistics |
| /quit      | Exit |

---

## Configuration

All settings in .env:

| Variable | Default | Description |
|----------|---------|-------------|
| GROQ_API_KEY | - | Required. Get at console.groq.com |
| GROQ_MODEL | llama-3.3-70b-versatile | LLM model |
| EMBEDDING_MODEL | all-MiniLM-L6-v2 | Embedding model |
| CHUNK_SIZE | 512 | Tokens per chunk |
| CHUNK_OVERLAP | 50 | Overlap between chunks |
| K_INIT | 20 | Initial ANN retrieval count |
| K_RERANK | 10 | Candidates sent to cross-encoder |
| K_FINAL | 5 | Final chunks used for context |
| SIMILARITY_THRESHOLD | 0.10 | Minimum cosine similarity to pass filter |
| MIN_QUALITY | 0.15 | Minimum quality before retrying |
| MAX_RETRIES | 3 | Max query rewrite attempts |
| TAVILY_API_KEY | - | Optional. For real web search |
| DEEP_RESEARCH_MAX_QUERIES | 5 | Max parallel queries in deep research |
| REDIS_HOST | localhost | Redis host |
| REDIS_PORT | 6379 | Redis port |
| CACHE_SIMILARITY_THRESHOLD | 0.95 | Semantic similarity for cache hit |
| REDIS_CACHE_TTL | 86400 | Cache entry TTL in seconds |

---

## Project Structure

```
rag_final/
  main.py                    entry point
  requirements.txt
  .env.example
  agent/
    orchestrator.py          THINK -> ACT -> OBSERVE -> REFINE loop
    tools.py                 all five tools
    web_search.py            async web search + deep research
    reasoning_agent.py       async Groq answer generation
    verifier_agent.py        async Groq hallucination check
  rag/
    retriever.py             LlamaIndex + ChromaDB + CrossEncoder pipeline
    context_builder.py       merge, dedup, compress
    cache.py                 L1 LRU + L2 Redis HNSW cache
    file_extractor.py        multi-format document parser
  memory/
    memory_manager.py        session memory + scratchpad
  utils/
    config.py                environment config
    logger.py                suppress third-party noise
```

---

## Dependencies

- groq: LLM API
- llama-index-core: vector index and chunking
- chromadb: persistent vector database
- sentence-transformers: HuggingFace embeddings + CrossEncoder reranking
- redis: optional semantic cache L2
- tavily-python: optional real web search
- aiohttp: async HTTP for web search
- pdfplumber, python-docx: document parsing

---

## Notes

- Run from inside the rag_final/ directory, not from a subdirectory
- First run downloads embedding model (~90MB) and cross-encoder (~65MB) automatically
- Redis is entirely optional — the system is fully functional without it
- TAVILY_API_KEY is optional — mock results are returned when not set
- Verified answers are cached; answers that fail verification are not cached
