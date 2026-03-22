# -*- coding: utf-8 -*-
"""
research_tools.py — Academic Paper Search for Orchestration Agents

Sources (no API key required, no extra pip):
  1. arXiv       — q-fin / math / physics / cs / stat pre-prints
  2. OpenAlex    — 214M published papers + citation counts (Nature, Science, etc.)
  3. Crossref    — DOI journal metadata, unlimited rate

Usage in orchestrate.py:
  from research_tools import search_papers, format_papers_for_prompt
  results = search_papers("path signature trading", fields=["qfin", "math"])
  context = format_papers_for_prompt(results, max_papers=5)
"""

import re
import time
import threading
from datetime import datetime
from typing import Optional
from xml.etree import ElementTree as ET

import requests

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "QuantTradingResearch/1.0",
    "Accept": "application/json",
})

ARXIV_FIELD_MAP = {
    "qfin":    ["q-fin.TR", "q-fin.MF", "q-fin.CP", "q-fin.PM", "q-fin.RM", "q-fin.ST"],
    "math":    ["math.PR", "math.OC", "math.ST", "math.NA", "math.FA"],
    "physics": ["physics.data-an", "cond-mat.stat-mech", "quant-ph"],
    "cs":      ["cs.LG", "cs.AI", "stat.ML", "cs.CE"],
    "all":     [],  # no category filter
}


# ─────────────────────────────────────────────────────────────
# 1. arXiv
# ─────────────────────────────────────────────────────────────
_ARXIV_LAST = 0.0
_ARXIV_LOCK = threading.Lock()

def _arxiv_search(query: str, categories: list[str], max_results: int = 8) -> list[dict]:
    """Query arXiv API. Returns plain-text abstract + PDF link."""
    global _ARXIV_LAST
    with _ARXIV_LOCK:
        gap = time.time() - _ARXIV_LAST
        if gap < 0.4:
            time.sleep(0.4 - gap)
        _ARXIV_LAST = time.time()

    if categories:
        cat_clause = "(" + " OR ".join(f"cat:{c}" for c in categories) + ")"
        full_query  = f"{cat_clause} AND all:{query}"
    else:
        full_query = f"all:{query}"

    try:
        r = _SESSION.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": full_query,
                "max_results":  max_results,
                "sortBy":       "submittedDate",
                "sortOrder":    "descending",
            },
            timeout=15,
        )
        r.raise_for_status()
    except Exception:
        return []

    ns = {
        "atom":       "http://www.w3.org/2005/Atom",
        "arxiv":      "http://arxiv.org/schemas/atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    }
    root = ET.fromstring(r.text)
    papers = []
    for entry in root.findall("atom:entry", ns):
        arxiv_id = entry.findtext("atom:id", "", ns).split("/abs/")[-1]
        pdf_url  = next(
            (lk.get("href") for lk in entry.findall("atom:link", ns)
             if lk.get("type") == "application/pdf"),
            f"https://arxiv.org/pdf/{arxiv_id}",
        )
        papers.append({
            "source":   "arXiv",
            "title":    (entry.findtext("atom:title", "", ns) or "").strip(),
            "authors":  [n.text for n in entry.findall("atom:author/atom:name", ns)][:4],
            "year":     (entry.findtext("atom:published", "", ns) or "")[:4],
            "abstract": (entry.findtext("atom:summary", "", ns) or "").strip()[:600],
            "arxiv_id": arxiv_id,
            "url":      f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url":  pdf_url,
            "doi":      entry.findtext("arxiv:doi", "", ns) or "",
            "citations": None,
            "categories": [c.get("term") for c in entry.findall("atom:category", ns)],
        })
    return papers


# ─────────────────────────────────────────────────────────────
# 2. OpenAlex
# ─────────────────────────────────────────────────────────────
def _reconstruct_abstract(inv: dict) -> str:
    """Reconstruct plain text from OpenAlex inverted-index abstract."""
    if not inv:
        return ""
    size = max(max(v) for v in inv.values() if v) + 1
    words = [""] * size
    for word, positions in inv.items():
        for p in positions:
            if p < size:
                words[p] = word
    return " ".join(words).strip()


def _openalex_search(query: str, year_from: int = 2020, max_results: int = 8) -> list[dict]:
    """Query OpenAlex API. Covers Nature, Science, all major journals."""
    try:
        r = _SESSION.get(
            "https://api.openalex.org/works",
            params={
                "search":   query,
                "per-page": max_results,
                "sort":     "-cited_by_count",
                "filter":   f"publication_year:>={year_from}",
                "select":   "id,title,publication_year,cited_by_count,"
                            "abstract_inverted_index,doi,best_oa_location,"
                            "authorships,concepts",
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    papers = []
    for w in data.get("results", []):
        abstract = _reconstruct_abstract(w.get("abstract_inverted_index") or {})
        oa_url   = (w.get("best_oa_location") or {}).get("landing_page_url") or ""
        papers.append({
            "source":     "OpenAlex",
            "title":      w.get("title") or "",
            "authors":    [a["author"]["display_name"]
                           for a in (w.get("authorships") or [])[:4]],
            "year":       w.get("publication_year"),
            "abstract":   abstract[:600],
            "doi":        w.get("doi") or "",
            "url":        oa_url or (f"https://doi.org/{w['doi']}" if w.get("doi") else ""),
            "pdf_url":    (w.get("best_oa_location") or {}).get("pdf_url") or "",
            "citations":  w.get("cited_by_count"),
            "concepts":   [c["display_name"] for c in (w.get("concepts") or [])[:5]],
            "arxiv_id":   "",
            "categories": [],
        })
    return papers


# ─────────────────────────────────────────────────────────────
# 3. Crossref
# ─────────────────────────────────────────────────────────────
def _crossref_search(query: str, max_results: int = 6) -> list[dict]:
    """Query Crossref. Best for published journal articles (Nature, Science, etc.)."""
    try:
        r = _SESSION.get(
            "https://api.crossref.org/works",
            params={"query": query, "rows": max_results,
                    "filter": "from-pub-date:2020-01-01"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    papers = []
    for w in data.get("message", {}).get("items", []):
        title   = (w.get("title") or [""])[0]
        journal = (w.get("container-title") or [""])[0]
        doi     = w.get("DOI") or ""
        year    = ((w.get("published-print") or w.get("published-online") or {})
                   .get("date-parts", [[0]])[0][0])
        authors = [f"{a.get('given','')} {a.get('family','')}".strip()
                   for a in (w.get("author") or [])[:4]]
        papers.append({
            "source":     f"Crossref/{journal}" if journal else "Crossref",
            "title":      title,
            "authors":    authors,
            "year":       year,
            "abstract":   "",   # Crossref doesn't return abstracts
            "doi":        doi,
            "url":        f"https://doi.org/{doi}" if doi else "",
            "pdf_url":    "",
            "citations":  w.get("is-referenced-by-count", 0),
            "concepts":   [],
            "arxiv_id":   "",
            "categories": [w.get("type", "")],
        })
    return papers


# ─────────────────────────────────────────────────────────────
# Unified Search
# ─────────────────────────────────────────────────────────────
def search_papers(
    query:       str,
    fields:      list[str] = None,
    year_from:   int = 2020,
    max_total:   int = 12,
    sources:     list[str] = None,
) -> list[dict]:
    """
    Parallel search across arXiv + OpenAlex + Crossref.

    Args:
        query:     Natural language query, e.g. "path signature trading strategy"
        fields:    Scope hint list — any of: "qfin", "math", "physics", "cs", "all"
                   Affects arXiv category filter. OpenAlex/Crossref search all fields.
        year_from: Only return papers from this year onward.
        max_total: Max papers returned (deduplicated).
        sources:   Subset of ["arxiv", "openalex", "crossref"]. Default: all three.

    Returns:
        Deduplicated list of paper dicts, sorted by citations (desc).
    """
    if fields is None:
        fields = ["qfin", "math", "cs"]
    if sources is None:
        sources = ["arxiv", "openalex", "crossref"]

    # Build arXiv categories from fields
    arxiv_cats: list[str] = []
    for f in fields:
        arxiv_cats.extend(ARXIV_FIELD_MAP.get(f, []))
    # If "all" in fields, clear category filter
    if "all" in fields:
        arxiv_cats = []

    results: dict[str, list] = {}
    lock     = threading.Lock()
    threads  = []

    def _run(key, fn, *args, **kwargs):
        try:
            papers = fn(*args, **kwargs)
        except Exception:
            papers = []
        with lock:
            results[key] = papers

    if "arxiv" in sources:
        t = threading.Thread(target=_run, daemon=True,
                             args=("arxiv", _arxiv_search, query, arxiv_cats, 8))
        threads.append(t)
    if "openalex" in sources:
        t = threading.Thread(target=_run, daemon=True,
                             args=("openalex", _openalex_search, query, year_from, 8))
        threads.append(t)
    if "crossref" in sources:
        t = threading.Thread(target=_run, daemon=True,
                             args=("crossref", _crossref_search, query, 6))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    # Merge + deduplicate by DOI or normalised title
    seen:    set[str] = set()
    merged:  list[dict] = []
    for key in ("arxiv", "openalex", "crossref"):
        for p in results.get(key, []):
            uid = p.get("doi") or re.sub(r"\W+", " ", (p.get("title") or "")).lower().strip()
            if uid and uid not in seen:
                seen.add(uid)
                merged.append(p)

    # Sort: prefer papers with abstract > citations > year
    def _rank(p):
        has_abstract = 1 if p.get("abstract") else 0
        cit          = p.get("citations") or 0
        yr           = p.get("year") or 0
        return (has_abstract, cit, yr)

    merged.sort(key=_rank, reverse=True)
    return merged[:max_total]


# ─────────────────────────────────────────────────────────────
# Formatter for agent prompt injection
# ─────────────────────────────────────────────────────────────
def format_papers_for_prompt(
    papers: list[dict],
    max_papers:   int = 6,
    include_abstract: bool = True,
) -> str:
    """
    Format paper list as a markdown block suitable for LLM prompt injection.

    Returns:
        Markdown string, or empty string if no papers found.
    """
    if not papers:
        return ""

    lines = ["## [Papers] Relevant Research\n"]
    for i, p in enumerate(papers[:max_papers], 1):
        title   = p.get("title") or "Untitled"
        authors = ", ".join(p.get("authors") or []) or "Unknown"
        year    = p.get("year") or "?"
        source  = p.get("source") or "?"
        url     = p.get("pdf_url") or p.get("url") or ""
        cit     = p.get("citations")
        abstract = (p.get("abstract") or "").strip()

        lines.append(f"### {i}. {title}")
        lines.append(f"**Source**: {source} | **Year**: {year}" +
                     (f" | **Citations**: {cit}" if cit is not None else ""))
        lines.append(f"**Authors**: {authors}")
        if url:
            lines.append(f"**Link**: {url}")
        if include_abstract and abstract:
            lines.append(f"\n> {abstract[:400]}{'...' if len(abstract) > 400 else ''}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Field-specific convenience wrappers
# ─────────────────────────────────────────────────────────────
def search_qfin(query: str, max_total: int = 10) -> list[dict]:
    """Search quantitative finance papers (q-fin.TR/MF/CP/RM + published journals)."""
    return search_papers(query, fields=["qfin", "cs"], max_total=max_total)

def search_math(query: str, max_total: int = 10) -> list[dict]:
    """Search mathematics papers (probability, stochastic calculus, optimization)."""
    return search_papers(query, fields=["math", "physics"], max_total=max_total)

def search_ml(query: str, max_total: int = 10) -> list[dict]:
    """Search ML/DL papers (cs.LG, stat.ML) + published journals."""
    return search_papers(query, fields=["cs", "qfin"], max_total=max_total)

def search_broad(query: str, max_total: int = 10) -> list[dict]:
    """Broad search across all fields via OpenAlex + Crossref only."""
    return search_papers(query, fields=["all"], sources=["openalex", "crossref"],
                         max_total=max_total)


# ─────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=== arXiv: path signature trading ===")
    papers = search_qfin("path signature trading strategy", max_total=5)
    print(format_papers_for_prompt(papers, max_papers=5))

    print("\n=== Math: fractional Brownian motion ===")
    papers = search_math("fractional Brownian motion stochastic calculus", max_total=5)
    print(format_papers_for_prompt(papers, max_papers=5))
