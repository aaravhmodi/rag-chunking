from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


RAW_ROOT = Path("data/raw")
BENCHMARK_ROOT = Path("data/benchmark")
DOCUMENTS_DIR = BENCHMARK_ROOT / "documents"
QUESTIONS_PATH = BENCHMARK_ROOT / "questions.jsonl"
METADATA_PATH = BENCHMARK_ROOT / "metadata.json"


def main() -> None:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, object] = {
        "schema_version": "1.0",
        "sources": {},
        "document_count": 0,
        "question_count": 0,
    }
    questions: list[dict[str, object]] = []
    document_counts: dict[str, int] = {}

    for dataset_name in ("scifact", "fiqa", "nfcorpus"):
        doc_count, dataset_questions, dataset_meta = normalize_beir_dataset(dataset_name)
        document_counts[f"beir:{dataset_name}"] = doc_count
        questions.extend(dataset_questions)
        metadata["sources"] |= {f"beir:{dataset_name}": dataset_meta}

    qasper_doc_count, qasper_questions, qasper_meta = normalize_qasper()
    document_counts["qasper"] = qasper_doc_count
    questions.extend(qasper_questions)
    metadata["sources"] |= {"qasper": qasper_meta}

    with QUESTIONS_PATH.open("w", encoding="utf-8") as handle:
        for row in questions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata["document_count"] = sum(document_counts.values())
    metadata["question_count"] = len(questions)
    metadata["document_counts"] = document_counts
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"documents": metadata["document_count"], "questions": metadata["question_count"]}, indent=2))


def normalize_beir_dataset(dataset_name: str) -> tuple[int, list[dict[str, object]], dict[str, object]]:
    root = RAW_ROOT / "beir" / dataset_name
    corpus_path = root / "corpus.jsonl"
    queries_path = root / "queries.jsonl"
    qrels_dir = root / "qrels"

    documents_written = 0
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        doc_id = f"beir:{dataset_name}:{row['_id']}"
        text_parts = [row.get("title", "").strip(), row.get("text", "").strip()]
        content = "\n\n".join(part for part in text_parts if part)
        DOCUMENTS_DIR.joinpath(f"{doc_id}.txt".replace(":", "__")).write_text(content, encoding="utf-8")
        documents_written += 1

    query_text_by_id: dict[str, str] = {}
    for line in queries_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        query_text_by_id[row["_id"]] = row["text"]

    questions: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}
    for qrel_path in sorted(qrels_dir.glob("*.tsv")):
        split_name = qrel_path.stem
        grouped_doc_ids: dict[str, list[str]] = defaultdict(list)
        with qrel_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if int(row["score"]) > 0:
                    grouped_doc_ids[row["query-id"]].append(f"beir:{dataset_name}:{row['corpus-id']}")

        for query_id, relevant_doc_ids in grouped_doc_ids.items():
            question_id = f"beir:{dataset_name}:{split_name}:{query_id}"
            questions.append(
                {
                    "question_id": question_id,
                    "question": query_text_by_id[query_id],
                    "answer": "",
                    "alternative_answers": [],
                    "source_doc": relevant_doc_ids[0],
                    "gold_evidence": "",
                    "relevant_doc_ids": relevant_doc_ids,
                    "question_type": "ir",
                    "difficulty": "unknown",
                    "metadata": {
                        "dataset": dataset_name,
                        "source": "beir",
                        "split": split_name,
                    },
                }
            )
        split_counts[split_name] = len(grouped_doc_ids)

    return documents_written, questions, {
        "source": "beir",
        "dataset": dataset_name,
        "documents": documents_written,
        "questions": len(questions),
        "splits": split_counts,
    }


def normalize_qasper() -> tuple[int, list[dict[str, object]], dict[str, object]]:
    split_to_path = {
        "train": RAW_ROOT / "qasper" / "qasper-train-v0.3.json",
        "validation": RAW_ROOT / "qasper" / "qasper-dev-v0.3.json",
        "test": RAW_ROOT / "qasper" / "qasper-test-v0.3.json",
    }

    questions: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}
    documents_written = 0

    for split_name, split_path in split_to_path.items():
        papers = json.loads(split_path.read_text(encoding="utf-8"))
        split_counts[split_name] = 0
        for paper_id, paper in papers.items():
            doc_id = f"qasper:{paper_id}"
            document_text = render_qasper_document(paper)
            DOCUMENTS_DIR.joinpath(f"{doc_id}.txt".replace(":", "__")).write_text(document_text, encoding="utf-8")
            documents_written += 1

            for qa in paper.get("qas", []):
                answers = qa.get("answers", [])
                canonical_answer, alternative_answers, gold_evidence = choose_qasper_answers(answers)
                evidence_start, evidence_end = find_span(document_text, gold_evidence)
                question_row = {
                    "question_id": f"qasper:{split_name}:{qa['question_id']}",
                    "question": qa["question"],
                    "answer": canonical_answer,
                    "alternative_answers": alternative_answers,
                    "source_doc": doc_id,
                    "gold_evidence": gold_evidence,
                    "relevant_doc_ids": [doc_id],
                    "evidence_start": evidence_start,
                    "evidence_end": evidence_end,
                    "question_type": "scientific_qa",
                    "difficulty": "unknown",
                    "metadata": {
                        "dataset": "qasper",
                        "source": "allenai",
                        "split": split_name,
                        "paper_id": paper_id,
                        "annotation_count": len(answers),
                    },
                }
                questions.append(question_row)
                split_counts[split_name] += 1

    return documents_written, questions, {
        "source": "allenai",
        "dataset": "qasper",
        "documents": documents_written,
        "questions": len(questions),
        "splits": split_counts,
    }


def render_qasper_document(paper: dict[str, object]) -> str:
    lines = [paper.get("title", "").strip(), "", paper.get("abstract", "").strip()]
    for section in paper.get("full_text", []):
        section_name = str(section.get("section_name", "")).strip()
        paragraphs = [paragraph.strip() for paragraph in section.get("paragraphs", []) if paragraph and paragraph.strip()]
        if section_name:
            lines.extend(["", section_name])
        if paragraphs:
            lines.extend(["", "\n\n".join(paragraphs)])
    return "\n".join(line for line in lines if line is not None).strip()


def choose_qasper_answers(answers: list[dict[str, object]]) -> tuple[str, list[str], str]:
    normalized_answers: list[str] = []
    evidence_candidates: list[str] = []

    for answer_row in answers:
        answer = answer_row.get("answer", {})
        text = ""
        extractive_spans = answer.get("extractive_spans") or []
        if extractive_spans:
            text = "; ".join(span.strip() for span in extractive_spans if span.strip())
        elif answer.get("free_form_answer"):
            text = str(answer["free_form_answer"]).strip()
        elif answer.get("yes_no") is True:
            text = "yes"
        elif answer.get("yes_no") is False:
            text = "no"
        if text:
            normalized_answers.append(text)

        evidence = answer.get("evidence") or answer.get("highlighted_evidence") or []
        for snippet in evidence:
            snippet_text = str(snippet).strip()
            if snippet_text:
                evidence_candidates.append(snippet_text)

    canonical_answer = normalized_answers[0] if normalized_answers else ""
    alternative_answers = []
    seen = {canonical_answer.lower()} if canonical_answer else set()
    for item in normalized_answers[1:]:
        lowered = item.lower()
        if lowered not in seen:
            seen.add(lowered)
            alternative_answers.append(item)
    gold_evidence = evidence_candidates[0] if evidence_candidates else ""
    return canonical_answer, alternative_answers, gold_evidence


def find_span(document_text: str, snippet: str) -> tuple[int | None, int | None]:
    if not snippet:
        return None, None
    start = document_text.find(snippet)
    if start >= 0:
        return start, start + len(snippet)

    normalized_document, index_map = normalize_with_index_map(document_text)
    normalized_snippet, _ = normalize_with_index_map(snippet)
    normalized_start = normalized_document.find(normalized_snippet)
    if normalized_start < 0:
        return None, None
    normalized_end = normalized_start + len(normalized_snippet)
    original_start = index_map[normalized_start]
    original_end = index_map[normalized_end - 1] + 1
    return original_start, original_end


def normalize_with_index_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    index_map: list[int] = []
    previous_was_space = False
    for index, char in enumerate(text):
        if char.isspace():
            if chars and not previous_was_space:
                chars.append(" ")
                index_map.append(index)
            previous_was_space = True
            continue
        chars.append(char)
        index_map.append(index)
        previous_was_space = False

    if chars and chars[-1] == " ":
        chars.pop()
        index_map.pop()
    return "".join(chars), index_map


if __name__ == "__main__":
    main()
