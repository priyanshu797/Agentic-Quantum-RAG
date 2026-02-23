"""
agent/web_search.py
Async web search: Tavily (real) or mock fallback.
Deep research fires multiple parallel queries via asyncio.gather.
"""
import asyncio
from typing import List, Dict, Any
from utils.config import cfg
from utils.logger import get_logger
logger = get_logger(__name__)

async def _tavily(query: str, max_results: int) -> List[Dict[str,str]]:
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.tavily.com/search",
                json={"api_key":cfg.TAVILY_API_KEY,"query":query,
                      "max_results":max_results,"search_depth":"advanced"},
                timeout=aiohttp.ClientTimeout(total=12)
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    return [{"title":res.get("title",""),"url":res.get("url",""),
                             "snippet":res.get("content","")} for res in data.get("results",[])]
    except Exception as e:
        logger.debug("Tavily failed: "+str(e))
    return []

def _mock(query: str) -> List[Dict[str,str]]:
    return [{"title":"Mock: "+query,"url":"https://example.com/mock",
             "snippet":"[MOCK] Set TAVILY_API_KEY in .env for real search results. Query: "+query}]

async def web_search(query: str, max_results: int = 5) -> List[Dict[str,str]]:
    if cfg.TAVILY_API_KEY:
        results = await _tavily(query, max_results)
        if results: return results
    return _mock(query)

async def deep_research(base_query: str, sub_queries: List[str]) -> Dict[str,Any]:
    """
    Fire base_query + sub_queries in parallel using asyncio.gather.
    Merge, deduplicate by URL, return structured research context.
    """
    all_queries = [base_query] + sub_queries[:cfg.DEEP_RESEARCH_MAX_QUERIES - 1]
    tasks       = [web_search(q, cfg.DEEP_RESEARCH_MAX_RESULTS) for q in all_queries]
    batches     = await asyncio.gather(*tasks, return_exceptions=True)

    seen_urls: set = set()
    merged:    list = []
    sources:   list = []

    for batch in batches:
        if isinstance(batch, Exception): continue
        for r in batch:
            url = r.get("url","")
            if url not in seen_urls:
                seen_urls.add(url); merged.append(r)
                if url not in sources: sources.append(url)

    context_parts = [
        "[Web "+str(i+1)+"] "+r.get("title","")+"\n"+r.get("snippet","")
        for i, r in enumerate(merged[:cfg.DEEP_RESEARCH_MAX_RESULTS])
    ]
    return {
        "context":      "\n\n".join(context_parts),
        "sources":      sources,
        "result_count": len(merged),
        "queries_used": all_queries,
    }
