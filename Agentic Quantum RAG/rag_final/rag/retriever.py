"""
rag/retriever.py
LlamaIndex + ChromaDB retrieval pipeline:
  Step 1: K_INIT candidates via ANN (LlamaIndex)
  Step 2: Threshold filter
  Step 3: CrossEncoder rerank -> K_RERANK
  Step 4: Return K_FINAL
"""
import os, logging, warnings
warnings.filterwarnings("ignore")
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utils.config import cfg
for n in ["sentence_transformers","transformers","chromadb","llama_index","httpx","urllib3","openai"]:
    logging.getLogger(n).setLevel(logging.ERROR)

class Retriever:
    def __init__(self):
        Settings.embed_model   = HuggingFaceEmbedding(model_name=cfg.EMBEDDING_MODEL)
        Settings.llm           = None
        Settings.chunk_size    = cfg.CHUNK_SIZE
        Settings.chunk_overlap = cfg.CHUNK_OVERLAP
        os.makedirs(cfg.CHROMA_PERSIST_DIR, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=cfg.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False))
        self._col    = self._chroma.get_or_create_collection(
            name="rag_docs", metadata={"hnsw:space":"cosine"})
        self._vstore = ChromaVectorStore(chroma_collection=self._col)
        self._sctx   = StorageContext.from_defaults(vector_store=self._vstore,
            docstore=SimpleDocumentStore(), index_store=SimpleIndexStore())
        self._index  = VectorStoreIndex.from_vector_store(self._vstore, storage_context=self._sctx)
        self._reranker: Optional[Any] = None
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception: pass

    def index_documents(self, documents: List[Dict[str,Any]]) -> int:
        splitter = SentenceSplitter(chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP)
        nodes = splitter.get_nodes_from_documents(
            [Document(text=d["text"],metadata=d.get("metadata",{})) for d in documents],
            show_progress=False)
        if not nodes: return 0
        self._index.insert_nodes(nodes)
        return len(nodes)

    def retrieve(self, query: str) -> List[Dict[str,Any]]:
        if self._col.count() == 0: return []
        nodes = self._index.as_retriever(similarity_top_k=cfg.K_INIT).retrieve(query)
        if not nodes: return []
        results = [{"text":n.node.get_content(),"metadata":n.node.metadata,
            "similarity":float(n.score) if n.score is not None else 0.5} for n in nodes]
        filtered = [r for r in results if r["similarity"] >= cfg.SIMILARITY_THRESHOLD] or results
        candidates = filtered[:cfg.K_RERANK]
        if self._reranker and len(candidates) > 1:
            scores = self._reranker.predict([[query,c["text"]] for c in candidates])
            for c,s in zip(candidates,scores): c["rerank_score"] = float(s)
            candidates = sorted(candidates, key=lambda x: x.get("rerank_score",0), reverse=True)
        return candidates[:cfg.K_FINAL]

    def count(self) -> int: return self._col.count()

    def reset(self):
        self._chroma.delete_collection("rag_docs")
        self._col    = self._chroma.get_or_create_collection(
            name="rag_docs", metadata={"hnsw:space":"cosine"})
        self._vstore = ChromaVectorStore(chroma_collection=self._col)
        self._sctx   = StorageContext.from_defaults(vector_store=self._vstore,
            docstore=SimpleDocumentStore(), index_store=SimpleIndexStore())
        self._index  = VectorStoreIndex.from_vector_store(self._vstore, storage_context=self._sctx)
