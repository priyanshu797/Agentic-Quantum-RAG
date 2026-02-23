"""
main.py - Agentic RAG Terminal
Run:  cd rag_final && python main.py
      python main.py --files doc.pdf notes.txt
      python main.py --reset
"""
import os, sys, uuid, warnings, asyncio
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging
for name in ["sentence_transformers","transformers","chromadb","httpx","urllib3",
             "llama_index","openai","filelock","huggingface_hub","torch","nltk","tqdm","redis","aiohttp"]:
    logging.getLogger(name).setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from argparse import ArgumentParser

from agent.orchestrator import OrchestratorAgent
from agent.reasoning_agent import ReasoningAgent
from agent.tools import (VectorRetrieverTool, WebSearchTool, DeepResearchTool,
                          MemoryTool, DirectAnswerTool, ToolRegistry)
from agent.verifier_agent import VerifierAgent
from memory.memory_manager import memory_manager
from rag.retriever import Retriever
from rag.cache import cache
from rag.file_extractor import load_files
from utils.config import cfg


def get_embedding(embedder, text: str) -> np.ndarray:
    return np.array(embedder.get_text_embedding(text), dtype=np.float32)


def ingest(paths, retriever):
    docs = load_files(paths)
    if not docs:
        print("No text found in files.")
        return 0
    total = retriever.index_documents(docs)
    for d in docs:
        print("Loaded: "+d["metadata"]["source"]+" ("+str(len(d["text"].split()))+" words)")
    print(str(total)+" chunks indexed.")
    print()
    return total


def main():
    parser = ArgumentParser(description="Agentic RAG Terminal")
    parser.add_argument("--files", nargs="*")
    parser.add_argument("--reset", action="store_true")
    args, _ = parser.parse_known_args()

    if not cfg.GROQ_API_KEY or cfg.GROQ_API_KEY == "your_groq_api_key_here":
        print("ERROR: Set GROQ_API_KEY in .env file.")
        sys.exit(1)

    retriever = Retriever()
    embedder  = HuggingFaceEmbedding(model_name=cfg.EMBEDDING_MODEL)

    if args.reset:
        retriever.reset()
        cache.flush()
        print("Reset complete.")
        print()

    registry = ToolRegistry()
    registry.register(VectorRetrieverTool(retriever))
    registry.register(WebSearchTool())
    registry.register(DeepResearchTool())
    registry.register(MemoryTool(memory_manager))
    registry.register(DirectAnswerTool())

    orchestrator    = OrchestratorAgent(registry)
    reasoning_agent = ReasoningAgent()
    verifier        = VerifierAgent()

    # Cache status
    stats = cache.stats()
    if stats.get("redis"):
        print("Cache: Redis connected  L1="+str(stats["l1_entries"])+" L2="+str(stats.get("l2_entries","?")))
    else:
        print("Cache: in-memory only (Redis not running — optional)")
    print("Vector DB: "+str(retriever.count())+" existing chunks")
    print("Web search: "+("Tavily (real)" if cfg.TAVILY_API_KEY else "mock (set TAVILY_API_KEY for real)"))
    print()

    if args.files:
        ingest(args.files, retriever)
    else:
        raw = input("Enter file path(s) (comma-separated) or press Enter to skip: ").strip()
        if raw:
            ingest([p.strip() for p in raw.split(",") if p.strip()], retriever)

    session = memory_manager.get_or_create(str(uuid.uuid4()))
    print("Commands: /files  /reset  /history  /cache  /quit")
    print()

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query: continue

        if query.lower() in ("quit","/quit","exit","q"): break

        if query.lower() == "/files":
            raw = input("File path(s): ").strip()
            if raw: ingest([p.strip() for p in raw.split(",") if p.strip()], retriever)
            continue

        if query.lower() == "/reset":
            retriever.reset(); cache.flush()
            session = memory_manager.get_or_create(str(uuid.uuid4()))
            print("Reset complete.\n"); continue

        if query.lower() == "/history":
            hist = session.history(n=20)
            if not hist:
                print("No history yet.\n")
            else:
                for m in hist:
                    print(("You" if m["role"]=="user" else "AI")+": "+m["content"][:120])
                print()
            continue

        if query.lower() == "/cache":
            print("Cache: "+str(cache.stats())+"\n"); continue

        session.add("user", query)
        has_docs   = retriever.count() > 0
        query_emb  = get_embedding(embedder, query)

        # ---- THINK -> ACT -> OBSERVE -> REFINE ----
        result    = orchestrator.run(query, session, query_emb=query_emb, has_docs=has_docs)
        context   = result["context"]
        sources   = result["sources"]
        tool_used = result["tool_used"]
        is_direct = tool_used == "direct_answer"
        hit       = result.get("cache_hit", False)
        iters     = result.get("iterations", 1)

        # ---- GENERATE ----
        gen    = reasoning_agent.generate(query, context, sources, session, is_direct)
        answer = gen["answer"]

        # ---- VERIFY + loop-back if failed ----
        verification = verifier.verify(query, context, answer)
        if not verification.get("passed") and context.strip():
            nudge        = context+"\n\nIMPORTANT: Be strictly factual. Only use the above context."
            gen          = reasoning_agent.generate(query, nudge, sources, session, is_direct)
            answer       = gen["answer"]
            verification = verifier.verify(query, context, answer)

        # ---- CACHE the final answer ----
        if not hit and context.strip() and verification.get("passed", True):
            cache.set(query, query_emb, answer, sources)

        session.add("assistant", answer, metadata={"sources":sources,"tool":tool_used})

        # ---- OUTPUT ----
        print()
        print("AI:")
        print(answer)
        print()

        if sources:
            print("Sources: "+", ".join(sources))

        grounded = verification.get("groundedness", 1.0)
        passed   = verification.get("passed", True)
        queries_used = result.get("queries_used", [])

        meta = ("[tool="+tool_used+" | iters="+str(iters)+
                " | K_init="+str(cfg.K_INIT)+
                " | K_rerank="+str(cfg.K_RERANK)+
                " | K_final="+str(cfg.K_FINAL)+
                " | grounded="+str(round(grounded,2))+
                " | verified="+str(passed)+
                " | cache_hit="+str(hit)+"")

        if queries_used:
            meta += " | deep_research_queries="+str(len(queries_used))

        meta += "]"
        print(meta)
        print()

if __name__ == "__main__":
    main()
