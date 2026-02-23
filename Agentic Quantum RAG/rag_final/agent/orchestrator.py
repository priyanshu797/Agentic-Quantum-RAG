"""
agent/orchestrator.py
Full agentic THINK -> ACT -> OBSERVE -> REFINE loop.

The agent:
  1. Classifies intent (conversational / factual / analytical / research)
  2. Selects the best tool
  3. Generates sub-queries for deep research if needed
  4. Evaluates output quality
  5. Rewrites query and retries if quality is low
  6. Returns best context found across all iterations
"""
import json, re, asyncio
from typing import Any, Dict, List
from groq import Groq
from agent.tools import ToolRegistry
from memory.memory_manager import Session
from utils.config import cfg
from utils.logger import get_logger
logger = get_logger(__name__)

_SYSTEM_PLAN = (
    "You are an intelligent agent router. The user may have uploaded documents.\n\n"
    "Available tools:\n{tools}\n\n"
    "DECISION RULES:\n"
    "1. vector_retriever  - ANY question about facts, topics, advice, summaries from uploaded documents\n"
    "2. web_search        - User explicitly asks about current/live/recent news or internet topics\n"
    "3. deep_research     - Complex multi-angle questions needing comprehensive research across many sources\n"
    "4. memory            - User refers to something said earlier in this conversation\n"
    "5. direct_answer     - ONLY for pure greetings: hi, hello, thanks, bye\n\n"
    "When documents are available, always prefer vector_retriever first.\n"
    "Use deep_research only for broad research questions like 'research everything about X'.\n\n"
    "For deep_research, also generate 3-5 specific sub-queries.\n\n"
    "Reply ONLY with valid JSON:\n"
    "{{\"tool\":\"name\",\"query\":\"optimised search text\",\"reasoning\":\"one line\","
    "\"sub_queries\":[\"q1\",\"q2\"]}}"
)

_SYSTEM_REWRITE = (
    "Rewrite this search query using different keywords or synonyms to improve retrieval results.\n"
    "Return ONLY the new query string, nothing else."
)

class OrchestratorAgent:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.client   = Groq(api_key=cfg.GROQ_API_KEY)

    def run(self, query: str, session: Session,
            query_emb=None, has_docs: bool = True) -> Dict[str,Any]:
        session.scratchpad.reset()
        best          = {"context":"","sources":[],"quality_score":0.0,"tool_used":"none"}
        current_query = query

        for it in range(1, cfg.MAX_RETRIES + 2):
            session.scratchpad.iterations = it

            # THINK: decide tool and optimise query
            plan        = self._think(current_query, session.history_str(4), it, has_docs)
            tool_name   = plan.get("tool","vector_retriever")
            opt_query   = plan.get("query", current_query)
            sub_queries = plan.get("sub_queries", [])
            reasoning   = plan.get("reasoning","")
            session.scratchpad.think(
                "["+str(it)+"] tool="+tool_name+" | reason="+reasoning)

            # ACT: run chosen tool
            tool = self.registry.get(tool_name) or self.registry.get("vector_retriever")
            inp  = {"query":opt_query,"session_id":session.session_id,
                    "embedding":query_emb,"sub_queries":sub_queries}
            result = tool.run(inp)
            session.scratchpad.record(tool_name, inp, result)

            # OBSERVE: evaluate quality
            quality = result.get("quality_score", 0.0)
            session.scratchpad.think(
                "["+str(it)+"] quality="+str(round(quality,2))+" success="+str(result.get("success")))

            if result.get("success") and quality > best["quality_score"]:
                best = {
                    "context":     result.get("context",""),
                    "sources":     result.get("sources",[]),
                    "quality_score": quality,
                    "tool_used":   tool_name,
                    "cache_hit":   result.get("cache_hit",False),
                    "queries_used":result.get("queries_used",[]),
                }

            # Stop conditions
            if result.get("direct"): break
            if quality >= cfg.MIN_QUALITY: break

            # REFINE: rewrite query for next iteration
            if it <= cfg.MAX_RETRIES:
                current_query = self._rewrite(query, current_query)
                session.scratchpad.think("Rewritten query: "+current_query)
            else:
                session.scratchpad.think("Max retries reached, using best result.")

        best["iterations"] = session.scratchpad.iterations
        best["thoughts"]   = session.scratchpad.thoughts
        return best

    def _think(self, query: str, history: str, it: int, has_docs: bool) -> Dict[str,Any]:
        context_note = ""
        if not has_docs:
            context_note = "\nNote: No documents are indexed. Prefer web_search or direct_answer."
        content = "User query: "+query
        if history: content = "History:\n"+history+"\n\n"+content
        if it > 1:  content += "\n\n(Retry "+str(it)+": use different keywords)"
        content += context_note
        try:
            r = self.client.chat.completions.create(
                model=cfg.GROQ_MODEL,
                messages=[
                    {"role":"system","content":_SYSTEM_PLAN.format(tools=self.registry.descriptions())},
                    {"role":"user","content":content},
                ],
                temperature=0.1, max_tokens=300,
            )
            return _parse(r.choices[0].message.content.strip())
        except Exception as e:
            logger.debug("_think failed: "+str(e))
            return {"tool":"vector_retriever","query":query,"sub_queries":[]}

    def _rewrite(self, original: str, previous: str) -> str:
        try:
            r = self.client.chat.completions.create(
                model=cfg.GROQ_MODEL,
                messages=[
                    {"role":"system","content":_SYSTEM_REWRITE},
                    {"role":"user","content":"Original: "+original+"\nPrevious attempt: "+previous},
                ],
                temperature=0.4, max_tokens=80,
            )
            return r.choices[0].message.content.strip().strip("\"'")
        except Exception:
            return original

def _parse(text: str) -> Dict[str,Any]:
    text = re.sub(r"```(?:json)?","",text).strip()
    try: return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except Exception: pass
    return {"tool":"vector_retriever","query":text,"sub_queries":[]}
