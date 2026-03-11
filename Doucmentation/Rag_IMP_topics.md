Important RAG Topics to Learn                                                                                                                                     
                                                                                                                                                                    
  ---                                                                                                                                                               
  1. Chunking Strategies (Beyond Fixed Size)                                                                                                                        
                                                                                                                                                                    
  What you have now is basic fixed-size chunking. Better approaches:                                                                                                
  - Semantic chunking — split on meaning, not character count                                                                                                       
  - Hierarchical chunking — store both full page AND sub-chunks, retrieve at both levels
  - Sentence window — embed single sentences but retrieve surrounding sentences for context

  ---
  2. Retrieval Strategies

  Basic similarity search is just the start:
  - Hybrid search — combine vector search + keyword search (Azure AI Search supports this natively with BM25)
  - MMR (Maximal Marginal Relevance) — avoids returning 5 chunks that all say the same thing
  - Re-ranking — after retrieving k chunks, use a second model to re-score and reorder them
  - Multi-query retrieval — generate 3-4 versions of the question, retrieve for each, merge results

  ---
  3. Evaluation

  How do you know if your RAG is actually good?
  - Faithfulness — is the answer grounded in the retrieved docs or hallucinated?
  - Answer relevance — does the answer actually address the question?
  - Context precision/recall — did you retrieve the right chunks?
  - Tools: RAGAs is the standard framework for this

  ---
  4. Indexing Pipeline Robustness

  - Incremental indexing — only re-index changed/new documents, not the whole corpus
  - Duplicate detection — avoid indexing same content twice
  - Document versioning — track when a doc was last indexed

  ---
  5. Query Understanding

  Before hitting the vector store:
  - Query rewriting — clean/expand the user's raw question
  - HyDE (Hypothetical Document Embedding) — generate a fake answer first, embed that, then search (often more accurate than embedding the question)
  - Query routing — decide which index/data source to search based on the question

  ---
  6. Advanced Metadata Filtering

  Instead of searching everything:
  "Find recipes from Famous old receipts.pdf only"
  "Search only pages 1-20"
  Azure AI Search supports pre-filtering by metadata before doing vector search — much faster and more precise.

  ---
  7. Hallucination Control

  - Grounding checks — verify the answer is actually supported by retrieved chunks
  - Citation verification — check that page numbers cited actually contain the claimed info
  - Confidence thresholds — if retrieval score is below X, say "I don't know" instead of guessing

  ---
  8. Production Concerns

  - Caching — cache embeddings of common queries to reduce API costs
  - Rate limiting — Azure OpenAI has TPM (tokens per minute) limits
  - Cost tracking — token usage per query adds up fast on large corpora
  - Async indexing — for large PDFs, index in background so app stays responsive

  ---
  9. Agentic RAG (Advanced)

  Beyond single-turn Q&A:
  - Self-correcting RAG — model checks its own answer and re-retrieves if unsatisfied
  - Multi-hop reasoning — answer requires combining info from multiple separate retrievals
  - Tool-use RAG — model decides whether to search, calculate, or use other tools

  ---
  Learning Priority

  Beginner  →  Chunking strategies, Hybrid search, Evaluation (RAGAs)
  Mid-level →  Re-ranking, Query rewriting, Metadata filtering, HyDE
  Advanced  →  Agentic RAG, Multi-hop, Self-correcting, Async pipelines

  You already have the foundation — chunking, embedding, vector store, generation. The next most impactful topic to learn would be Evaluation with RAGAs because it
  tells you objectively whether any improvement you make is actually working.


  Rag_IMP_topics.md outlines the learning roadmap beyond your current
  implementation:
  - Beginner: Chunking strategies, Hybrid search, RAGAs evaluation
  - Mid-level: Re-ranking, HyDE, Query rewriting, Metadata filtering
  - Advanced: Agentic RAG, Multi-hop reasoning, Self-correcting RAG
