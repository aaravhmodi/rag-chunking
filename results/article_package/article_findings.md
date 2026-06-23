# Article Findings

## Main Claim
On the scoped QASPER test slice, paragraph chunking provides the strongest overall retrieval quality and the best evidence-span coverage, while fixed-128 is the fastest but least evidence-preserving strategy.

## Evidence
- Paragraph has the highest Recall@k, MRR, answer exact match, and evidence-span recall.
- Adaptive is the closest competitor but trails paragraph on every quality-facing metric.
- Sentence-level chunking retains decent recall but almost collapses on evidence-span coverage.
- Fixed-128 minimizes latency but fragments evidence most aggressively.

## Interpretation
The diagnostics suggest that the main failure mode remains missing the relevant material altogether, but chunk boundaries still matter: paragraph chunking converts more relevant hits into full evidence-span hits than the more aggressive splitters.
