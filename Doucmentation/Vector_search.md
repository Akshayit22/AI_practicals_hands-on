# Vector DB Search Strategies

---

## 1. Similarity Search
> Basic — already in your code

```python
db.similarity_search(query, k=5)
```

| | |
|---|---|
| **How** | Converts query → embedding → finds nearest vectors (cosine/dot product) |
| **Best for** | Simple factual questions |
| **Weakness** | Only looks at vector distance, ignores keyword matches |

---

## 2. MMR — Maximal Marginal Relevance

```python
db.max_marginal_relevance_search(query, k=5, fetch_k=20, lambda_mult=0.5)
```

| | |
|---|---|
| **How** | Fetches `fetch_k=20` candidates, then picks `k=5` that are relevant but diverse |
| **`lambda_mult`** | `0` = max diversity, `1` = max relevance |
| **Best for** | Word docs with repetitive content (same info in multiple sections) |
| **Weakness** | Slightly slower than plain similarity |

---

## 3. Hybrid Search (Semantic + Keyword)

```
Vector similarity score  +  BM25 keyword score  =  final rank
```

```python
# Azure AI Search — hybrid is default when both are configured
vector_store.similarity_search(query, search_type="hybrid")
```

| | |
|---|---|
| **How** | Combines vector similarity with BM25 keyword scoring |
| **Best for** | Documents with specific terms, names, codes, table values |
| **Weakness** | Needs search backend support (not available in FAISS) |
| **Supported by** | Azure AI Search, OpenSearch |

---

## 4. Self-Query / Metadata Filtering

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever

retriever = SelfQueryRetriever.from_llm(
    llm, vector_store, document_description, metadata_field_info
)
retriever.invoke("Find HR policy from page 3")
```

| | |
|---|---|
| **How** | LLM extracts filters from the query (e.g. `page=3`, `source="policy.pdf"`) |
| **Best for** | PDF RAG — filters by page, source file, doc type |
| **Weakness** | Needs an LLM call per query |

---

## 5. Contextual Compression

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=db.as_retriever(search_kwargs={"k": 10})
)
```

| | |
|---|---|
| **How** | Retrieves `k=10` chunks → LLM compresses each to only the relevant sentence |
| **Best for** | Large chunks (like PDF pages) where only 1-2 lines are relevant |
| **Weakness** | Extra LLM calls = higher cost |

---

## 6. Multi-Query Retrieval

```python
from langchain.retrievers import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(), llm=llm
)
```

| | |
|---|---|
| **How** | LLM rephrases your query 3-5 ways → runs all → merges results |
| **Best for** | Ambiguous queries, when users ask vague questions |
| **Weakness** | 3-5x more embedding calls |

---

## Which Strategy to Use

| Your Data | Best Strategy |
|---|---|
| PDF (policy doc) | Hybrid + Metadata Filter (by page/source) |
| Excel/CSV (food reviews) | Similarity + MMR (avoid repetitive reviews) |
| Word doc (mixed content) | Hybrid + Contextual Compression |
| Vague user queries | Multi-Query |
| Large chunks | Contextual Compression |

---

## Recommended Pipeline

```
User Query
    ↓
Multi-Query (rephrase)       ← handles vague queries
    ↓
Hybrid Search                ← semantic + keyword
    ↓
MMR re-ranking               ← removes duplicate results
    ↓
Contextual Compression       ← trim irrelevant parts
    ↓
LLM Answer Generation
```

---

## YouTube Search Queries

### General
```
RAG retrieval strategies LangChain 2024
advanced RAG techniques explained
RAG beyond naive retrieval
Advanced RAG from scratch
```

### Per Strategy

| Strategy | Search Query |
|---|---|
| Similarity Search | `vector similarity search FAISS LangChain` |
| MMR | `maximal marginal relevance LangChain retriever` |
| Hybrid Search | `hybrid search BM25 vector search LangChain` |
| Self-Query | `self query retriever LangChain metadata filtering` |
| Contextual Compression | `contextual compression retriever LangChain` |
| Multi-Query | `multi query retriever LangChain` |

### Best Channels
- **LangChain** — official channel, tutorials on all retrievers
- **Sam Witteveen** — deep RAG videos
- **Prompt Engineering** — advanced RAG techniques
- **1littlecoder** — practical implementations
