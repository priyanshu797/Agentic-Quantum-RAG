"""
agent/verifier_agent.py
Async critic agent: checks groundedness, hallucination, conflicts.
Triggers loop-back regeneration if answer fails validation.
"""
import asyncio, json, re
from typing import Dict, Any
from groq import AsyncGroq
from utils.config import cfg
from utils.logger import get_logger
logger = get_logger(__name__)

_SYSTEM = (
    "You are a strict fact-checker for AI-generated answers.\n"
    "Given QUESTION, CONTEXT, and ANSWER — evaluate carefully.\n\n"
    "Reply ONLY with valid JSON:\n"
    "{\"groundedness\":<0.0-1.0>,\"responsiveness\":<0.0-1.0>,"
    "\"hallucination\":<true|false>,\"conflicts\":<true|false>,\"reason\":\"brief\"}"
)

async def _verify_async(query: str, context: str, answer: str) -> Dict[str,Any]:
    client = AsyncGroq(api_key=cfg.GROQ_API_KEY)
    msg    = "QUESTION:\n"+query+"\n\nCONTEXT:\n"+context[:1800]+"\n\nANSWER:\n"+answer
    try:
        r = await client.chat.completions.create(
            model=cfg.GROQ_MODEL,
            messages=[{"role":"system","content":_SYSTEM},{"role":"user","content":msg}],
            temperature=0.1, max_tokens=150,
        )
        return _parse(r.choices[0].message.content.strip())
    except Exception as e:
        logger.debug("verify failed: "+str(e))
        return _ok("Verification skipped.")
    finally:
        await client.close()

class VerifierAgent:
    def verify(self, query: str, context: str, answer: str) -> Dict[str,Any]:
        if not context.strip(): return _ok("Direct answer — no context to verify.")
        try:
            result = asyncio.run(_verify_async(query, context, answer))
        except Exception:
            return _ok("Verification error.")
        passed = (
            result.get("groundedness",1.0)   >= 0.5
            and result.get("responsiveness",1.0) >= 0.5
            and not result.get("hallucination",False)
            and not result.get("conflicts",False)
        )
        result["passed"] = passed
        return result

def _ok(reason: str) -> Dict[str,Any]:
    return {"groundedness":1.0,"responsiveness":1.0,"hallucination":False,
            "conflicts":False,"reason":reason,"passed":True}

def _parse(text: str) -> Dict[str,Any]:
    text = re.sub(r"```(?:json)?","",text).strip()
    try: return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except Exception: pass
    return {"groundedness":0.7,"responsiveness":0.7,"hallucination":False,
            "conflicts":False,"reason":"parse failed"}
