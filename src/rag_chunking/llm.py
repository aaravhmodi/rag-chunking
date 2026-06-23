from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from urllib import request

from rag_chunking.evaluation import is_answerable
from rag_chunking.models import LLMGrade, QuestionExample, RetrievalResult


def _build_context(results: list[RetrievalResult]) -> str:
    return "\n\n".join(result.chunk.text for result in results)


def _extractive_answer(question: QuestionExample, results: list[RetrievalResult]) -> str:
    if not results:
        return ""
    context = _build_context(results)
    candidates = [question.answer, *question.alternative_answers]
    for candidate in candidates:
        if candidate and candidate.lower() in context.lower():
            return candidate
    first_sentence = re.split(r"(?<=[.!?])\s+", results[0].chunk.text.strip())[0]
    return first_sentence[:280]


class LLMJudge:
    name = "heuristic"

    def grade(self, question: QuestionExample, results: list[RetrievalResult]) -> LLMGrade:
        raise NotImplementedError


class HeuristicJudge(LLMJudge):
    name = "heuristic"

    def grade(self, question: QuestionExample, results: list[RetrievalResult]) -> LLMGrade:
        generated_answer = _extractive_answer(question, results)
        context = _build_context(results).lower()
        answer_score = 0.0
        if is_answerable(question):
            candidates = [question.answer, *question.alternative_answers]
            if any(candidate and candidate.lower() in context for candidate in candidates):
                answer_score = 1.0
        hallucination_score = 0.0 if not generated_answer else (0.0 if generated_answer.lower() in context else 1.0)
        return LLMGrade(
            answer_score=answer_score,
            hallucination_score=hallucination_score,
            generated_answer=generated_answer,
            rationale="Heuristic judge uses extractive grounding against retrieved context.",
        )


@dataclass(slots=True)
class OpenAIJudge(LLMJudge):
    model: str = "gpt-4.1-mini"
    api_key: str = ""
    name: str = "openai"

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def grade(self, question: QuestionExample, results: list[RetrievalResult]) -> LLMGrade:
        context = _build_context(results)
        generated_answer = _extractive_answer(question, results)
        prompt = {
            "question": question.question,
            "reference_answer": question.answer,
            "alternative_answers": question.alternative_answers,
            "retrieved_context": context,
            "candidate_answer": generated_answer,
            "task": (
                "Return JSON with keys answer_score, hallucination_score, generated_answer, rationale. "
                "Set answer_score to 1 when the candidate answer is correct, otherwise 0. "
                "Set hallucination_score to 1 when the candidate answer is unsupported by retrieved context, otherwise 0."
            ),
        }
        payload = json.dumps(
            {
                "model": self.model,
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": json.dumps(prompt)}],
                    }
                ],
            }
        ).encode("utf-8")
        req = request.Request(
            "https://api.openai.com/v1/responses",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=60) as response:  # pragma: no cover - network dependent
            body = json.loads(response.read().decode("utf-8"))
        text = _response_text(body)
        parsed = json.loads(text)
        return LLMGrade(
            answer_score=float(parsed.get("answer_score", 0.0)),
            hallucination_score=float(parsed.get("hallucination_score", 0.0)),
            generated_answer=str(parsed.get("generated_answer", generated_answer)),
            rationale=str(parsed.get("rationale", "")),
        )


def _response_text(body: dict) -> str:
    for item in body.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                return text
    return "{}"


def build_llm_judge(spec: str | None) -> LLMJudge | None:
    if not spec:
        return None
    normalized = spec.strip().lower()
    if normalized in {"heuristic", "local"}:
        return HeuristicJudge()
    if normalized.startswith("openai"):
        parts = spec.split(":", maxsplit=1)
        model = parts[1] if len(parts) == 2 and parts[1] else "gpt-4.1-mini"
        return OpenAIJudge(model=model, name=f"openai:{model}")
    raise ValueError(f"Unsupported LLM judge backend: {spec}")
