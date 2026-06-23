# Benchmarking Notes

## Current State

The benchmark now includes:

- lexical, embedding, and hybrid retriever backends behind one interface
- semantic chunking driven by sentence embeddings
- optional LLM answer grading and hallucination checks
- paired significance testing across datasets and metrics
- JPEG-only article plots plus JPEG table renders for reporting

The original sparse retriever and chunking-only setup is still supported, but it is no longer the only path through the pipeline.

## What Was Slow

The original lexical retriever compared every query against every chunk.

Runtime therefore scaled approximately with:

- number of strategies
- number of questions
- number of chunks across the corpus

This was acceptable for `data/sample` and impractical for `QASPER` and mixed `BEIR` runs.

## What Changed

### Step 1: Replace brute-force chunk scanning

The retriever in `src/rag_chunking/retrieval.py` now builds:

- token frequency vectors per chunk
- per-chunk token lengths
- document frequencies
- an inverted index from token to matching chunk IDs

At query time it only scores chunks that share at least one token with the query.

### Step 2: Upgrade the sparse ranking function

The ranking function is now BM25-style sparse scoring instead of cosine similarity over raw token counts.
This improves the retrieval baseline itself while keeping the benchmark architecture simple and dependency-light.

### Step 3: Add safer run controls

The CLI now supports:

- `--question-dataset`
- `--question-split`
- `--max-documents`
- `--max-questions`
- `--cache-dir`
- `--diagnostics-output`

These flags are intended to make exploratory and article-oriented runs reproducible without launching accidental all-corpus experiments.

### Step 4: Add chunk caching

Chunking is deterministic for a given:

- strategy
- ordered document IDs
- document text content

When `--cache-dir` is set, the pipeline stores chunk outputs on disk keyed by a content fingerprint of the loaded document collection plus the strategy name.
Repeated runs with the same filtered document set can therefore skip chunk rebuilding entirely.

### Step 5: Add per-question diagnostics

When `--diagnostics-output` is set, the benchmark writes one row per question per strategy with:

- first relevant rank
- first evidence rank
- top-1 document ID
- top-1 relevance flag
- answer exact match
- evidence-span coverage
- coarse failure mode

This is the main artifact for debugging why one chunking strategy wins or loses.

### Step 6: Preserve mixed benchmark support

The benchmark schema supports both:

- retrieval-only datasets such as `BEIR`
- answer/evidence datasets such as `QASPER`

Question rows can therefore include:

- relevant document IDs
- exact answers
- alternative answers
- evidence spans when available

### Step 7: Add retriever backends

The retriever interface now supports:

- `lexical` for BM25-style sparse ranking
- `embedding` for semantic ranking through an embedding backend
- `hybrid` for score fusion between lexical and embedding retrieval

The embedding layer currently defaults to a deterministic hash backend so the project remains runnable offline. Optional sentence-transformers and OpenAI embeddings are supported through the same interface.

### Step 8: Add semantic chunking

The chunker interface now includes a semantic sentence-grouping mode that uses sentence embeddings to decide whether adjacent sentences should stay together.

This gives the article a direct comparison between:

- fixed-size splitting
- sentence and paragraph grouping
- adaptive paragraph heuristics
- semantic grouping driven by embeddings

### Step 9: Add answer grading and hallucination checks

The benchmark can now score retrieved context with either:

- a heuristic local judge
- an OpenAI-backed judge

The per-question diagnostics now include:

- generated answer
- answer score
- hallucination score

These fields are intended for article tables and failure analysis, not as a replacement for human evaluation.

### Step 10: Add significance testing

The pipeline can now compare strategies with paired randomization tests across datasets and metrics.

This is meant to support article claims that go beyond descriptive ranking tables.

## Recommended Run Order

1. Run a small sanity check on `data/sample`.
2. Run one benchmark dataset and one split.
3. Start with `paragraph`, `adaptive`, and `semantic`.
4. Compare `lexical` against `hybrid` and `embedding` retriever backends.
5. Enable LLM grading only after the retrieval outputs look sane.
6. Generate article artifacts only after the scoped run looks reasonable.

## Example Commands

### Fast smoke test

```powershell
rag-benchmark `
  --documents data/sample/documents `
  --questions data/sample/questions.jsonl `
  --strategies fixed-128 paragraph
```

### Scoped QASPER test run

```powershell
rag-benchmark `
  --documents data/benchmark/documents `
  --questions data/benchmark/questions.jsonl `
  --question-dataset qasper `
  --question-split test `
  --strategies paragraph adaptive `
  --cache-dir .cache/chunks `
  --top-k 5 `
  --max-documents 250 `
  --max-questions 300 `
  --output results/qasper_scope/results.json `
  --csv-output results/qasper_scope/results.csv `
  --report-output results/qasper_scope/report.md `
  --plots-dir results/qasper_scope/plots
```

### Mixed-dataset article slice

```powershell
rag-benchmark `
  --documents data/benchmark/documents `
  --questions data/benchmark/questions.jsonl `
  --question-split test `
  --strategies paragraph adaptive `
  --top-k 5 `
  --max-questions 1000
```

## Remaining Limits

- This is still a lightweight in-process sparse retriever rather than a production search engine.
- The first cached run still pays the initial chunking cost.
- Full mixed-corpus runs may still be slow when document counts and chunk counts are both large.
- Embedding and LLM backends depend on optional extras or API access.
- Significance testing is paired and descriptive; it is not a substitute for a full experimental design.

The next major speed upgrade would be caching or persisting embedding indexes in addition to chunks.
