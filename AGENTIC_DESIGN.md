# Agentic Design


## Pre Processing

### Chunking
- Done at ingestion in parser.py — strategy depends on doc type, not a uniform splitter.
- compromis — regex split on section headers (LE(S) VENDEUR(S), L(ES) ACQUEREUR(S), ARTICLE N —) → ~15 chunks per doc.
- dpe — split on all-caps lines ≥ 5 chars → ~5 chunks per doc.
identite / domicile — always one chunk per doc (short single-topic docs gain nothing from splitting).
- Fallback cascade (compromis/dpe only, if headers match nothing): OCR block boundaries → whole document as one chunk.
- Each chunk carries a stable, human-readable ID (dossier_1/compromis#VENDEUR) used as the citation unit, plus mean OCR confidence.
- Rationale: section-level chunks give precise citations and better targeted retrieval; revisit (add overlap) only if cross-section context gets lost.
### Hybrid search
- Lives in hybrid_search() in engine.py, called only by the search_documents tool. Three stages:
1. Parallel retrieval — cosine similarity (normalized embeddings, numpy dot product) and BM25 each independently rank the same filtered chunk set.
2. RRF fusion — top-15 from each list, unioned, scored 1/(60 + cosine_rank) + 1/(60 + bm25_rank) (60 is the standard constant from the RRF paper).
3. Cross-encoder rerank — mmarco-mMiniLMv2 scores each (query, chunk_text) pair jointly, re-sorts, returns top-5.
relevance_score = sigmoid of the cross-encoder logit (0–1), better calibrated than raw cosine; the best score per request becomes the retrieval_relevance metric.
Why hybrid: cosine alone misses exact lexical matches (article numbers, proper names, legal codes) — BM25 catches those.
Cost: cross-encoder adds ~50–150 ms on CPU; acceptable at 76 chunks, and the swap-to-lighter-model path is noted if latency ever matters.
Supports optional metadata filters (dossier, doc_type) applied before ranking.

### Embedding
Embeddings give the system semantic search — the ability to match a query to chunks by meaning rather than by exact words. That matters here for a few specific reasons:

- Users don't phrase questions in the document's words. A notary asks "quel est le prix de vente?" but the compromis might say "moyennant le prix principal de 285 000 euros". There's no shared keyword to match on — but in embedding space, the query vector and that chunk's vector land close together because they're about the same thing.
- The corpus is OCR'd, so exact text is unreliable. Scanned documents contain corruption ("CARTE MATIONALE"). Keyword matching breaks on garbled tokens; embeddings degrade gracefully because the surrounding context still places the chunk in roughly the right semantic neighborhood.
- French morphology and paraphrase. "Vendeur", "le cédant", "la partie venderesse" — a multilingual embedding model maps these near each other without anyone maintaining a synonym list.

## Pattern: ReWoo (Reasoning Without Observation)

### Why ReWoo over ReAct

**ReAct** interleaves reasoning and tool calls in a loop — each step can adapt based on what the previous step found:

```
Thought: I need to check all documents in dossier 3
Action: get_dossier_documents(3)
Observation: [results]
Thought: I see a corrupted CNI, let me verify against the compromis
Action: search_documents("FONTAINE", dossier=3)
Observation: [results]
Final answer: ...
```

It's adaptive but it's a true loop — the runaway depth problem lives here.

**ReWoo** produces the full plan upfront in one pass, executes all tools (potentially in parallel), then synthesizes:

```
Plan:    [get_document_inventory(), get_dossier_documents(3)]
Execute: both tools run in parallel
Solve:   synthesize final answer from all results
```

Always exactly 3 LLM calls regardless of complexity. The loop depth problem is solved **architecturally** — there is no loop.

### Why ReWoo fits this use case

Most of our query types have a predictable retrieval strategy — "Y a-t-il des incohérences dans le dossier 2?" always maps to `get_dossier_documents(2)`. The planner can determine the right tools upfront without needing intermediate results. The documents are well-defined and the question types are predictable.

### ReWoo flow

```
User query
    │
    ▼
Planner LLM — produces a plan (list of tool calls + parameters)
    │
    ▼
Execute all tools (in parallel where possible)
    │
    ▼
Solver LLM — synthesizes final answer from all tool results + source citations
```

### Tradeoff vs ReAct

| | ReWoo | ReAct |
|---|---|---|
| LLM calls | Always 3 (plan, execute, solve) | Variable (1 per tool + 1 final) |
| Latency | Lower (parallel tool execution) | Higher (sequential) |
| Cost | Predictable | Variable |
| Adaptive discovery | No — plan is fixed | Yes — each step informs the next |
| Loop risk | None | Requires max_iterations guard |

ReAct is the upgrade path for more open-ended query patterns where intermediate results change the retrieval strategy.

---

## Tools

| Tool | Signature | Returns | When the planner uses it |
|---|---|---|---|
| `search_documents` | `(query, dossier?, doc_type?)` | Top-k chunks ranked by hybrid search (cosine + BM25 → RRF → cross-encoder), each with `id`, `text`, `ocr_confidence`, `relevance_score` | Targeted lookups where a specific piece of information is needed — price, address, date, expiry |
| `get_dossier_documents` | `(dossier)` | All chunks from that dossier, unranked | Coherence checks, cross-document inconsistency detection, questions that require reading the whole dossier |
| `get_document_inventory` | `()` | Per-dossier map of present doc types, `missing_types`, `complete` flag, and the completeness checklist | "Pièces manquantes" questions, completeness audits |

### How retrieval quality flows through

`search_documents` returns a `relevance_score` per chunk (sigmoid-normalised cross-encoder logit, 0–1). The chat route takes the highest score across all `search_documents` calls as the request's `retrieval_relevance` metric. When only `get_dossier_documents` is used, `retrieval_relevance` defaults to `1.0` — all documents were explicitly retrieved, so there is no ranking uncertainty.

### Completeness checklist

`get_document_inventory` always returns the following checklist alongside the per-dossier data, so the solver can reason about what is absent without an extra prompt:

| Required | Doc type |
|---|---|
| 1 per dossier | `compromis` de vente |
| 1 per party | `identite` — pièce d'identité valide |
| 1 per party | `domicile` — justificatif de moins de 3 mois |
| 1 per dossier | `dpe` — Diagnostic de Performance Énergétique |

The `missing_types` field in each dossier entry lists doc types from this set that have no corresponding profile, giving the solver a direct answer without requiring it to infer absence from a list.

---

## Agent Loop

ReWoo eliminates the loop depth problem architecturally — the planner produces a bounded plan in one shot and there is no back-and-forth loop. The only remaining guard needed is a cap on the number of tool calls the planner is allowed to include in its plan (to prevent an overly ambitious plan from burning excessive tokens):

`max_tools_per_plan = 5`

This is a simpler and more principled constraint than a runtime iteration counter.

---

## No Regex in the Query Pipeline

Using regex to detect "dossier 1" in a query is fragile and redundant:
- "le premier dossier", "dossier numéro 1" would silently fail
- The LLM handles French naturally via tool parameters

Regex is only acceptable at ingestion time for classifying filenames into document types — that is structural, not linguistic.

---

## Source Citation

Each chunk has a stable, human-readable ID encoding its exact location:

```
dossier_1/compromis#VENDEUR
dossier_3/scan_006#block_1
```

Tool results include IDs alongside text. The system prompt instructs the LLM to always cite which IDs it used. Example output:

> "Le vendeur du dossier 1 est Jean-Pierre MOREAU, né le 15/03/1958 à Paris.
> **Sources:** `dossier_1/compromis#VENDEUR`, `dossier_1/scan_id_001#IDENTITE`"

For low OCR confidence chunks, the agent is instructed to flag this explicitly:

> "La CNI de M. FONTAINE (scan_006) présente une qualité OCR faible (confiance: 0.41) — les informations peuvent être inexactes."
