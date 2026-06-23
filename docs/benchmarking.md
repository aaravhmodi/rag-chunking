# Benchmarking Notes

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
- precomputed chunk norms
- an inverted index from token to matching chunk IDs

At query time it only scores chunks that share at least one token with the query.

### Step 2: Keep evaluation semantics stable

The scoring function is still cosine similarity over sparse token counts.
The optimization changes retrieval efficiency, not the benchmark metric definitions.

### Step 3: Add safer run controls

The CLI now supports:

- `--question-dataset`
- `--question-split`
- `--max-documents`
- `--max-questions`

These flags are intended to make exploratory and article-oriented runs reproducible without launching accidental all-corpus experiments.

### Step 4: Preserve mixed benchmark support

The benchmark schema supports both:

- retrieval-only datasets such as `BEIR`
- answer/evidence datasets such as `QASPER`

Question rows can therefore include:

- relevant document IDs
- exact answers
- alternative answers
- evidence spans when available

## Recommended Run Order

1. Run a small sanity check on `data/sample`.
2. Run one benchmark dataset and one split.
3. Start with `paragraph` and `adaptive`.
4. Add more aggressive chunking baselines after confirming runtime.
5. Generate article artifacts only after the scoped run looks reasonable.

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

- This is still a simple lexical retriever, not BM25.
- Chunking is recomputed per strategy and per run.
- Full mixed-corpus runs may still be slow when document counts and chunk counts are both large.

The next major speed upgrade would be chunk caching plus a BM25-style sparse index.
