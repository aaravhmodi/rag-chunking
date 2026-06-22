# RAG Chunking Benchmark Report

## Abstract

This report evaluates how chunking strategy changes retrieval effectiveness and efficiency in a dependency-light RAG benchmark. Strategies are compared on recall, ranking quality, answer coverage, chunk count, chunk length, and retrieval latency.

## Dataset

- Documents: 2
- Questions: 4
- Average document length (tokens): 68.0
- Question types: causal, definition, factual

## Method

Each strategy chunks the same document collection, indexes chunk text with a lexical retriever, and retrieves the top-k chunks for each question. Relevance is counted when the correct source document is retrieved and the chunk contains either the gold evidence string or the answer string.

## Results

| Strategy | Recall@k | MRR | nDCG@k | Answer EM | Avg chunks/doc | Avg chunk chars | Chunking ms | Retrieval ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sliding-window | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 | 455.00 | 0.050 | 0.079 |
| sentence | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 | 455.00 | 0.082 | 0.083 |
| adaptive | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 | 457.00 | 0.419 | 0.087 |
| paragraph | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 | 457.00 | 0.068 | 0.089 |
| fixed-256 | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 | 455.00 | 0.051 | 0.096 |
| fixed-128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 | 455.00 | 0.115 | 0.147 |

## Findings

- Highest recall@k: `fixed-128` at 1.000.
- Best MRR: `fixed-128` at 1.000.
- Lowest retrieval latency: `sliding-window` at 0.079 ms.
- Lowest average chunk count per document: `fixed-128` at 1.00.
- Strategy spread: chunk count ranges from 1.00 to 1.00, and average chunk length ranges from 455.00 to 457.00 characters.

## Limitations

- Retrieval is lexical rather than embedding-based, so semantic recall is under-measured.
- Relevance uses exact string containment for evidence and answers, which is stricter than human judgment.
- Results are descriptive; this scaffold does not run significance tests.

## Reproducibility

- Install with `python -m pip install -e .`.
- Run `rag-benchmark` with the same document, question, and strategy arguments used for this report.