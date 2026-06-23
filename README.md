# RAG Chunking Benchmark

MVP framework for evaluating how chunking strategy affects retrieval quality, latency, and index size in a RAG-style pipeline.

## What is implemented

- Multiple chunking strategies:
  - fixed-size
  - sentence-based
  - paragraph-based
  - sliding-window
  - adaptive heuristic
- Document and QA dataset loaders
- Lightweight sparse lexical retriever with BM25-style chunk ranking
- Evaluation metrics:
  - Recall@k
  - MRR
  - nDCG@k
  - answer exact match
  - latency and index statistics
- CLI entrypoint for running experiments

This scaffold is intentionally dependency-light so it runs in an empty workspace. It gives you a working benchmark harness now, and a clean place to plug in sentence-transformers, FAISS, Chroma, or OpenAI embeddings later.

## Dataset format

Documents live in a directory:

```text
data/sample/documents/
  october_crisis.txt
  rag_intro.txt
```

Questions live in a JSONL file:

```json
{"question":"Which group kidnapped James Cross?","answer":"FLQ","source_doc":"october_crisis","gold_evidence":"members of the FLQ kidnapped British diplomat James Cross","question_type":"factual"}
```

Required fields:

- `question`
- `answer`
- `source_doc`
- `gold_evidence`

Optional fields:

- `question_type`
- `metadata`

## Install

```powershell
python -m pip install -e .
```

## Run

```powershell
rag-benchmark `
  --documents data/sample/documents `
  --questions data/sample/questions.jsonl `
  --strategies fixed-128 fixed-256 paragraph sentence adaptive `
  --top-k 5
```

For larger benchmarks, start with a scoped run:

```powershell
rag-benchmark `
  --documents data/benchmark/documents `
  --questions data/benchmark/questions.jsonl `
  --question-dataset qasper `
  --question-split test `
  --strategies paragraph adaptive `
  --top-k 5 `
  --max-documents 250 `
  --max-questions 300
```

To generate article-style artifacts in one run:

```powershell
rag-benchmark `
  --documents data/sample/documents `
  --questions data/sample/questions.jsonl `
  --strategies fixed-128 fixed-256 paragraph sentence adaptive sliding-window `
  --top-k 5 `
  --output results/results.json `
  --csv-output results/results.csv `
  --report-output results/report.md `
  --plots-dir results/plots
```

## Performance Notes

- The retriever now uses an inverted index and BM25-style sparse scoring, so it only scores chunks that share tokens with the query.
- Full benchmark runs can still be expensive on large corpora because chunking long documents produces many chunk candidates.
- Recommended workflow:
  1. Start with one dataset and one split.
  2. Run `paragraph` and `adaptive` first.
  3. Add `fixed-128` and `sentence` after you have confirmed runtime is acceptable.
  4. Use `--max-documents` and `--max-questions` for exploratory runs.

Detailed pipeline notes live in [docs/benchmarking.md](docs/benchmarking.md).

## Current gaps

1. Add embedding backends behind the retriever interface.
2. Add semantic chunking using sentence embeddings.
3. Add LLM-based answer grading and hallucination checks.
4. Add significance testing across multiple benchmark datasets.
