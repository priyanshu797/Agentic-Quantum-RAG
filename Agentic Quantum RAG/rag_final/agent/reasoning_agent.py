"""
agent/reasoning_agent.py
Async Groq LLM call for grounded answer generation.
Output is always formatted as bullet points / numbered lines, not paragraphs.
"""
import asyncio
from typing import Dict, Any, List
from groq import AsyncGroq
from memory.memory_manager import Session
from utils.config import cfg
from utils.logger import get_logger
logger = get_logger(__name__)

_SYSTEM = (
    "You are a precise, grounded AI assistant.\n\n"
    "RULES:\n"
    "- Answer using ONLY the provided Context.\n"
    "- If the answer is not in the Context, say exactly: "
    "'This information is not available in the provided documents.'\n"
    "- Never invent or assume facts.\n"
    "- Format your answer as clear bullet points or numbered lines.\n"
    "- Each distinct fact or point goes on its own separate line.\n"
    "- Do NOT write paragraphs. One idea per line.\n"
    "- Be concise and direct.\n"
    "- Mention source filenames when referencing specific information."
)

_PROMPT = (
    "Context:\n{context}\n\n"
    "Conversation history:\n{history}\n\n"
    "Question: {query}\n\n"
    "Answer (bullet points, one fact per line):"
)

_DIRECT = (
    "Conversation history:\n{history}\n\n"
    "Question: {query}\n\nAnswer:"
)

async def _call_groq_async(messages: List[Dict], max_tokens: int = 700) -> str:
    client = AsyncGroq(api_key=cfg.GROQ_API_KEY)
    try:
        r = await client.chat.completions.create(
            model=cfg.GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        logger.debug("Async Groq call failed: "+str(e))
        return "Error generating response. Please try again."
    finally:
        await client.close()

class ReasoningAgent:
    def generate(self, query: str, context: str, sources: List[str],
                 session: Session, is_direct: bool = False) -> Dict[str,Any]:
        """Synchronous entry point — runs async Groq call in event loop."""
        history = session.history_str(n=6)
        if is_direct or not context.strip():
            msg = _DIRECT.format(history=history or "None", query=query)
        else:
            msg = _PROMPT.format(context=context, history=history or "None", query=query)

        messages = [
            {"role":"system","content":_SYSTEM},
            {"role":"user","content":msg},
        ]

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _call_groq_async(messages))
                    answer = future.result()
            else:
                answer = loop.run_until_complete(_call_groq_async(messages))
        except Exception:
            answer = asyncio.run(_call_groq_async(messages))

        return {"answer": answer, "sources": sources, "model": cfg.GROQ_MODEL}
