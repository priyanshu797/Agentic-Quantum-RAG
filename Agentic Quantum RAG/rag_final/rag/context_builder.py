import nltk
from typing import List, Dict, Any
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

class ContextBuilder:
    def __init__(self, max_tokens: int = 1500):
        self.max_tokens = max_tokens

    def build(self, chunks: List[Dict[str,Any]]) -> Dict[str,Any]:
        if not chunks: return {"context":"","sources":[],"token_count":0}
        chunks = sorted(chunks, key=lambda c: c.get("rerank_score",c.get("similarity",0)), reverse=True)
        seen, parts, sources, tokens = set(), [], [], 0
        for c in chunks:
            src = c.get("metadata",{}).get("source","unknown")
            if src not in sources: sources.append(src)
            unique = [s.strip() for s in nltk.sent_tokenize(c["text"])
                      if s.strip().lower() not in seen and len(s.strip()) > 10
                      and not seen.add(s.strip().lower())]
            if not unique: continue
            chunk_text = " ".join(unique)
            chunk_tok  = len(chunk_text.split())
            if tokens + chunk_tok > self.max_tokens:
                remaining = self.max_tokens - tokens
                if remaining > 20:
                    parts.append("["+src+"] "+" ".join(chunk_text.split()[:remaining]))
                break
            parts.append("["+src+"] "+chunk_text)
            tokens += chunk_tok
        return {"context":"\n\n".join(parts),"sources":sources,"token_count":tokens}
