"""
agent/tools.py
All tools. VectorRetriever uses cache. WebSearch + DeepResearch are async, run via asyncio.run().
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
from rag.retriever import Retriever
from rag.context_builder import ContextBuilder
from rag.cache import cache
from memory.memory_manager import MemoryManager
from utils.config import cfg

class BaseTool(ABC):
    name: str
    description: str
    @abstractmethod
    def run(self, inp: Dict[str,Any]) -> Dict[str,Any]: pass

class VectorRetrieverTool(BaseTool):
    name        = "vector_retriever"
    description = ("Semantic search over indexed documents. "
                   "Use for ANY question about uploaded files, facts, topics, or information.")
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.ctx       = ContextBuilder(max_tokens=1500)

    def run(self, inp: Dict[str,Any]) -> Dict[str,Any]:
        query = inp.get("query","").strip()
        emb   = inp.get("embedding")
        if not query:
            return {"success":False,"error":"Empty query","context":"","sources":[],"quality_score":0.0}
        if emb is not None:
            hit = cache.get(query, emb)
            if hit:
                resp, srcs = hit
                return {"success":True,"context":resp,"sources":srcs,
                        "quality_score":1.0,"cache_hit":True}
        chunks = self.retriever.retrieve(query)
        if not chunks:
            return {"success":False,"error":"No relevant chunks found",
                    "context":"","sources":[],"quality_score":0.0}
        built   = self.ctx.build(chunks)
        quality = sum(c.get("rerank_score",c.get("similarity",0)) for c in chunks) / len(chunks)
        return {"success":True,"context":built["context"],"sources":built["sources"],
                "token_count":built["token_count"],"quality_score":round(quality,3),"cache_hit":False}

class WebSearchTool(BaseTool):
    name        = "web_search"
    description = ("Search the web for live, current, or recent information not in documents. "
                   "Use when user asks about news, current events, or topics not in uploaded files.")
    def run(self, inp: Dict[str,Any]) -> Dict[str,Any]:
        from agent.web_search import web_search
        query   = inp.get("query","")
        results = asyncio.run(web_search(query, max_results=5))
        context = "\n\n".join("[Web "+str(i+1)+"] "+r.get("title","")+"\n"+r.get("snippet","")
                               for i,r in enumerate(results))
        return {"success":True,"context":context,
                "sources":[r.get("url","") for r in results],"quality_score":0.6}

class DeepResearchTool(BaseTool):
    name        = "deep_research"
    description = ("Multi-query parallel web research for complex topics requiring broad coverage. "
                   "Use when a question requires comprehensive research across multiple angles.")
    def run(self, inp: Dict[str,Any]) -> Dict[str,Any]:
        from agent.web_search import deep_research
        base_query  = inp.get("query","")
        sub_queries = inp.get("sub_queries",[])
        result      = asyncio.run(deep_research(base_query, sub_queries))
        return {"success":True,"context":result["context"],"sources":result["sources"],
                "quality_score":0.7,"queries_used":result["queries_used"],
                "result_count":result["result_count"]}

class MemoryTool(BaseTool):
    name        = "memory"
    description = "Returns recent conversation history. Use when user refers to a previous message."
    def __init__(self, mgr: MemoryManager):
        self.mgr = mgr
    def run(self, inp: Dict[str,Any]) -> Dict[str,Any]:
        session = self.mgr.get(inp.get("session_id",""))
        if not session:
            return {"success":True,"context":"","sources":[],"quality_score":0.8}
        return {"success":True,"context":"Conversation history:\n"+session.history_str(6),
                "sources":[],"quality_score":0.8}

class DirectAnswerTool(BaseTool):
    name        = "direct_answer"
    description = "Use ONLY for greetings and pure chitchat (hi, hello, thanks, bye). No retrieval."
    def run(self, inp: Dict[str,Any]) -> Dict[str,Any]:
        return {"success":True,"context":"","sources":[],"quality_score":1.0,"direct":True}

class ToolRegistry:
    def __init__(self):
        self._t: Dict[str,BaseTool] = {}
    def register(self, tool: BaseTool):
        self._t[tool.name] = tool
    def get(self, name: str) -> Optional[BaseTool]:
        return self._t.get(name)
    def names(self) -> List[str]:
        return list(self._t.keys())
    def descriptions(self) -> str:
        return "\n".join("- "+t.name+": "+t.description for t in self._t.values())
