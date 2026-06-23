# RAG Chunking Benchmark Report (QASPER Scoped Test, Four Methods)

## Abstract

This report evaluates how chunking strategy changes retrieval effectiveness and efficiency in a dependency-light RAG benchmark. Strategies are compared on recall, ranking quality, answer coverage, evidence-span coverage, chunk count, chunk length, and retrieval latency. All tables, figures, and summary statements are computed directly from the experiment runs included in this package.

## Dataset

- Documents: 250
- Questions: 233
- Average document length (tokens): 4061.0
- Question types: scientific_qa
- Datasets: qasper=233
- Splits: test=233

## Method

Each strategy chunks the same document collection, indexes chunk text with a lexical retriever, and retrieves the top-k chunks for each question. Relevance is counted when the correct source document is retrieved and the chunk contains either the gold evidence string or the answer string. For questions with annotated character spans, evidence-span recall@k counts whether any retrieved chunk fully covers the labeled evidence span.

## Experimental Results

| Strategy | Recall@k | MRR | nDCG@k | Answer EM | Evidence R@k | Avg chunks/doc | Avg chunk chars | Chunking ms | Retrieval ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| paragraph | 0.412 | 0.328 | 0.619 | 0.264 | 0.205 | 9.32 | 2758.58 | 0.000 | 14.251 |
| adaptive | 0.403 | 0.317 | 0.614 | 0.233 | 0.176 | 15.38 | 1672.02 | 0.000 | 15.989 |
| sentence | 0.403 | 0.305 | 0.612 | 0.251 | 0.005 | 17.23 | 1490.27 | 0.000 | 8.258 |
| fixed-128 | 0.386 | 0.310 | 0.621 | 0.194 | 0.000 | 31.61 | 811.03 | 0.000 | 5.379 |

## Findings

- Highest recall@k: `paragraph` at 0.412.
- Best MRR: `paragraph` at 0.328.
- Highest evidence-span recall@k: `paragraph` at 0.205.
- Lowest retrieval latency: `fixed-128` at 5.379 ms.
- Lowest average chunk count per document: `paragraph` at 9.32.
- Strategy spread: chunk count ranges from 9.32 to 31.61, and average chunk length ranges from 811.03 to 2758.58 characters.

## Slice Analysis

### dataset=qasper

| Strategy | Recall@k | MRR | nDCG@k | Answer EM | Evidence R@k | Avg chunks/doc | Avg chunk chars | Chunking ms | Retrieval ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| paragraph | 0.412 | 0.328 | 0.619 | 0.264 | 0.205 | 9.32 | 2758.58 | 0.000 | 13.603 |
| adaptive | 0.403 | 0.317 | 0.614 | 0.233 | 0.176 | 15.38 | 1672.02 | 0.000 | 9.111 |
| sentence | 0.403 | 0.305 | 0.612 | 0.251 | 0.005 | 17.23 | 1490.27 | 0.000 | 7.395 |
| fixed-128 | 0.386 | 0.310 | 0.621 | 0.194 | 0.000 | 31.61 | 811.03 | 0.000 | 4.137 |

### split=test

| Strategy | Recall@k | MRR | nDCG@k | Answer EM | Evidence R@k | Avg chunks/doc | Avg chunk chars | Chunking ms | Retrieval ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| paragraph | 0.412 | 0.328 | 0.619 | 0.264 | 0.205 | 9.32 | 2758.58 | 0.000 | 13.230 |
| adaptive | 0.403 | 0.317 | 0.614 | 0.233 | 0.176 | 15.38 | 1672.02 | 0.000 | 9.231 |
| sentence | 0.403 | 0.305 | 0.612 | 0.251 | 0.005 | 17.23 | 1490.27 | 0.000 | 7.448 |
| fixed-128 | 0.386 | 0.310 | 0.621 | 0.194 | 0.000 | 31.61 | 811.03 | 0.000 | 4.611 |


## Diagnostics

### adaptive

- Failure modes: missed_evidence_span=51, missed_relevant=139, success_evidence=36, success_relevant=7
- Sample missed evidence questions:
  - `qasper:test:85e417231a4bbb6691f7a89bd81710525f8fec4c` first relevant rank=3, first evidence rank=None, top1 doc=`qasper:1605.05156`
  - `qasper:test:2974237446d04da33b78ce6d22a477cdf80877b7` first relevant rank=3, first evidence rank=None, top1 doc=`qasper:1605.05156`
  - `qasper:test:ec8f39d32084996ab825debd7113c71daac38b06` first relevant rank=4, first evidence rank=None, top1 doc=`qasper:1609.06791`
  - `qasper:test:fcdbaa08cccda9968f3fd433c99338cc60f596a7` first relevant rank=3, first evidence rank=None, top1 doc=`qasper:1611.02988`
  - `qasper:test:fc436a4f3674e42fb280378314bfe77ba0c99f2e` first relevant rank=2, first evidence rank=None, top1 doc=`qasper:1611.03599`

### fixed-128

- Failure modes: missed_evidence_span=84, missed_relevant=143, success_relevant=6
- Sample missed evidence questions:
  - `qasper:test:fcdbaa08cccda9968f3fd433c99338cc60f596a7` first relevant rank=3, first evidence rank=None, top1 doc=`qasper:1608.08738`
  - `qasper:test:d0b967bfca2039c7fb05b931c8b9955f99a468dc` first relevant rank=4, first evidence rank=None, top1 doc=`qasper:1610.08815`
  - `qasper:test:0778cbbd093f8b779f7cf26302b2a8e081ccfb40` first relevant rank=1, first evidence rank=None, top1 doc=`qasper:1703.10152`
  - `qasper:test:a4b77a20e067789691e0ab246bc5b11913d77ae1` first relevant rank=1, first evidence rank=None, top1 doc=`qasper:1703.04009`
  - `qasper:test:4cf05da602669a4c09c91ff5a1baae6e30adefdf` first relevant rank=2, first evidence rank=None, top1 doc=`qasper:1701.09123`

### paragraph

- Failure modes: missed_evidence_span=47, missed_relevant=137, success_evidence=42, success_relevant=7
- Sample missed evidence questions:
  - `qasper:test:2974237446d04da33b78ce6d22a477cdf80877b7` first relevant rank=3, first evidence rank=None, top1 doc=`qasper:1605.05134`
  - `qasper:test:0f9c1586f1b4b531fa4fd113e767d06af90b1ae8` first relevant rank=5, first evidence rank=None, top1 doc=`qasper:1706.01678`
  - `qasper:test:fcdbaa08cccda9968f3fd433c99338cc60f596a7` first relevant rank=2, first evidence rank=None, top1 doc=`qasper:1611.02988`
  - `qasper:test:fc436a4f3674e42fb280378314bfe77ba0c99f2e` first relevant rank=3, first evidence rank=None, top1 doc=`qasper:1611.03599`
  - `qasper:test:d0b967bfca2039c7fb05b931c8b9955f99a468dc` first relevant rank=3, first evidence rank=None, top1 doc=`qasper:1604.00125`

### sentence

- Failure modes: missed_evidence_span=87, missed_relevant=139, success_evidence=1, success_relevant=6
- Sample missed evidence questions:
  - `qasper:test:ec8f39d32084996ab825debd7113c71daac38b06` first relevant rank=2, first evidence rank=None, top1 doc=`qasper:1611.03599`
  - `qasper:test:fcdbaa08cccda9968f3fd433c99338cc60f596a7` first relevant rank=3, first evidence rank=None, top1 doc=`qasper:1608.08738`
  - `qasper:test:fc436a4f3674e42fb280378314bfe77ba0c99f2e` first relevant rank=4, first evidence rank=None, top1 doc=`qasper:1611.03599`
  - `qasper:test:d0b967bfca2039c7fb05b931c8b9955f99a468dc` first relevant rank=5, first evidence rank=None, top1 doc=`qasper:1610.08815`
  - `qasper:test:6e134d51a795c385d72f38f36bca4259522bcf51` first relevant rank=2, first evidence rank=None, top1 doc=`qasper:1606.01404`


## Limitations

- Retrieval is lexical rather than embedding-based, so semantic recall is under-measured.
- Relevance uses exact string containment for evidence and answers, which is stricter than human judgment.
- Results are descriptive; this scaffold does not run significance tests.

## Reproducibility

- Install with `python -m pip install -e .`.
- Run `rag-benchmark` with the same document, question, and strategy arguments used for this report.
