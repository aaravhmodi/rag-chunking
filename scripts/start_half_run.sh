#!/usr/bin/env bash
set -euo pipefail

cd /home/ec2-user/rag-chunking-full
mkdir -p results/half

nohup python3.12 -m rag_chunking.cli \
  --documents data/benchmark/documents \
  --questions data/benchmark/questions.jsonl \
  --strategies paragraph adaptive sentence fixed-128 \
  --top-k 5 \
  --cache-dir .cache/chunks \
  --max-documents 34019 \
  --output results/half/results.json \
  > results/half/run.log 2>&1 < /dev/null &

echo $! > results/half/run.pid
echo "started $(cat results/half/run.pid)"
