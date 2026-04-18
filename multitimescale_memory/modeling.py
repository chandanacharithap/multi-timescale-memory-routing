from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass

from .types import EpisodeInput, SupportDoc


def _truncate(text: str, limit: int = 1600) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]


@dataclass(slots=True)
class AnswerResult:
    answer: str
    confidence: float
    latency: float


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    family: str
    size_tier: str
    backend: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "frozen-parametric": ModelSpec("frozen-parametric", "frozen", "toy", "frozen"),
    "google/flan-t5-small": ModelSpec("google/flan-t5-small", "flan-t5", "small", "hf_seq2seq"),
    "google/flan-t5-base": ModelSpec("google/flan-t5-base", "flan-t5", "mid", "hf_seq2seq"),
    "google/flan-ul2": ModelSpec("google/flan-ul2", "flan", "large", "hf_seq2seq"),
    "Qwen/Qwen2.5-1.5B-Instruct": ModelSpec(
        "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen2.5",
        "mid",
        "hf_causal",
    ),
    "Qwen/Qwen2.5-7B-Instruct": ModelSpec(
        "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5",
        "large",
        "hf_causal",
    ),
    "HuggingFaceTB/SmolLM2-360M-Instruct": ModelSpec(
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "smollm2",
        "small",
        "hf_causal",
    ),
}


def resolve_model_spec(model_name: str) -> ModelSpec:
    return MODEL_SPECS.get(
        model_name,
        ModelSpec(model_name, model_name.split("/")[0] if "/" in model_name else "custom", "unknown", "hf_seq2seq"),
    )


class FrozenParametricModel:
    def __init__(self) -> None:
        self._knowledge: dict[tuple[str, str], str] = {}

    def clone(self) -> "FrozenParametricModel":
        other = FrozenParametricModel()
        other._knowledge = dict(self._knowledge)
        return other

    def seed(self, subject: str, relation: str, answer: str) -> None:
        self._knowledge[(subject, relation)] = answer

    def answer_parametric(self, episode: EpisodeInput) -> AnswerResult:
        answer = episode.parametric_answer
        if answer is None:
            answer = self._knowledge.get((episode.subject, episode.relation), "unknown")
        return AnswerResult(answer=answer, confidence=episode.parametric_confidence, latency=0.2)

    def answer_with_evidence(self, episode: EpisodeInput, docs: list[SupportDoc]) -> AnswerResult:
        if docs:
            best = max(docs, key=lambda doc: (doc.trust + doc.relevance, doc.timestamp))
            if best.answer:
                return AnswerResult(answer=best.answer, confidence=max(best.trust, 0.5), latency=0.5)
        return self.answer_parametric(episode)

    def answer(self, episode: EpisodeInput) -> tuple[str, float]:
        result = self.answer_parametric(episode)
        return result.answer, result.confidence


class HuggingFaceQAModel:
    _MODEL_CACHE: dict[str, tuple[object, object]] = {}
    _GENERATION_CACHE: dict[str, dict[str, AnswerResult]] = {}

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        max_new_tokens: int = 24,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def clone(self) -> "HuggingFaceQAModel":
        return self

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        cached = self._MODEL_CACHE.get(self.model_name)
        if cached is not None:
            self._tokenizer, self._model = cached
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for HuggingFaceQAModel. "
                "Install it or use model_name='frozen-parametric' for dependency-light journal infrastructure tests."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.eval()
        self._MODEL_CACHE[self.model_name] = (self._tokenizer, self._model)

    def _generate(self, prompt: str) -> AnswerResult:
        self._ensure_loaded()
        prompt_cache = self._GENERATION_CACHE.setdefault(self.model_name, {})
        cached = prompt_cache.get(prompt)
        if cached is not None:
            return cached
        started = time.perf_counter()
        assert self._tokenizer is not None
        assert self._model is not None
        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=min(getattr(self._tokenizer, "model_max_length", 512), 512),
        )
        outputs = self._model.generate(
            **encoded,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        answer = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        latency = time.perf_counter() - started
        confidence = 0.4 if answer else 0.0
        result = AnswerResult(answer=answer or "unknown", confidence=confidence, latency=latency)
        prompt_cache[prompt] = result
        return result

    def answer_parametric(self, episode: EpisodeInput) -> AnswerResult:
        prompt = (
            "Answer the question with a short phrase.\n"
            f"Question: {episode.question}\n"
            "Answer:"
        )
        return self._generate(prompt)

    def answer_with_evidence(self, episode: EpisodeInput, docs: list[SupportDoc]) -> AnswerResult:
        evidence_lines = []
        for doc in docs:
            evidence_lines.append(f"[{doc.doc_id}] {_truncate(doc.text)}")
        joined = "\n".join(evidence_lines) if evidence_lines else "No evidence."
        prompt = (
            "Use the evidence to answer the question with a short phrase. "
            "If the evidence is insufficient, answer 'unknown'.\n"
            f"Question: {episode.question}\n"
            f"Evidence:\n{joined}\n"
            "Answer:"
        )
        return self._generate(prompt)


class HuggingFaceCausalQAModel:
    _MODEL_CACHE: dict[str, tuple[object, object]] = {}
    _GENERATION_CACHE: dict[str, dict[str, AnswerResult]] = {}

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 32,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def clone(self) -> "HuggingFaceCausalQAModel":
        return self

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        cached = self._MODEL_CACHE.get(self.model_name)
        if cached is not None:
            self._tokenizer, self._model = cached
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for HuggingFaceCausalQAModel. "
                "Install it or use model_name='frozen-parametric' for dependency-light tests."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.eval()
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._MODEL_CACHE[self.model_name] = (self._tokenizer, self._model)

    def _generate(self, prompt: str) -> AnswerResult:
        self._ensure_loaded()
        prompt_cache = self._GENERATION_CACHE.setdefault(self.model_name, {})
        cached = prompt_cache.get(prompt)
        if cached is not None:
            return cached
        started = time.perf_counter()
        assert self._tokenizer is not None
        assert self._model is not None
        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=min(getattr(self._tokenizer, "model_max_length", 1024), 1024),
        )
        prompt_length = int(encoded["input_ids"].shape[-1])
        outputs = self._model.generate(
            **encoded,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        decoded = self._tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
        answer = decoded.splitlines()[0].strip() if decoded else "unknown"
        latency = time.perf_counter() - started
        confidence = 0.35 if answer else 0.0
        result = AnswerResult(answer=answer or "unknown", confidence=confidence, latency=latency)
        prompt_cache[prompt] = result
        return result

    def answer_parametric(self, episode: EpisodeInput) -> AnswerResult:
        prompt = (
            "Answer the question with a short phrase.\n"
            f"Question: {episode.question}\n"
            "Answer:"
        )
        return self._generate(prompt)

    def answer_with_evidence(self, episode: EpisodeInput, docs: list[SupportDoc]) -> AnswerResult:
        evidence_lines = []
        for doc in docs:
            evidence_lines.append(f"[{doc.doc_id}] {_truncate(doc.text)}")
        joined = "\n".join(evidence_lines) if evidence_lines else "No evidence."
        prompt = (
            "Use the evidence to answer the question with a short phrase. "
            "If the evidence is insufficient, answer 'unknown'.\n"
            f"Question: {episode.question}\n"
            f"Evidence:\n{joined}\n"
            "Answer:"
        )
        return self._generate(prompt)


def model_runtime_status(model_name: str) -> dict[str, object]:
    spec = resolve_model_spec(model_name)
    if spec.backend == "frozen":
        return {
            "model_name": model_name,
            "backend": spec.backend,
            "loadable": True,
            "missing_dependencies": [],
        }

    required = ["transformers", "torch"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    return {
        "model_name": model_name,
        "backend": spec.backend,
        "loadable": not missing,
        "missing_dependencies": missing,
    }


def assert_models_available(model_names: list[str], attempt_load: bool = False) -> list[dict[str, object]]:
    statuses = [model_runtime_status(model_name) for model_name in model_names]
    missing = [status for status in statuses if not status["loadable"]]
    if missing:
        details = ", ".join(
            f"{status['model_name']} missing {','.join(status['missing_dependencies'])}"
            for status in missing
        )
        raise RuntimeError(
            "Model runtime dependencies are missing for the requested journal run: "
            f"{details}. Install the missing packages before running real Hugging Face models."
        )
    if attempt_load:
        for model_name in model_names:
            spec = resolve_model_spec(model_name)
            if spec.backend == "frozen":
                continue
            if spec.backend == "hf_causal":
                model = HuggingFaceCausalQAModel(model_name=model_name)
            else:
                model = HuggingFaceQAModel(model_name=model_name)
            model._ensure_loaded()
    return statuses


def build_model(model_name: str):
    spec = resolve_model_spec(model_name)
    if spec.backend == "frozen":
        return FrozenParametricModel()
    if spec.backend == "hf_causal":
        return HuggingFaceCausalQAModel(model_name=model_name)
    return HuggingFaceQAModel(model_name=model_name)
