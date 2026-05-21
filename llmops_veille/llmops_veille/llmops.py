"""
llmops.py — Module LLMOps pour VeilleTech
==========================================
Fournit 4 composants :
  - LLMTracer       : context manager qui logue chaque appel LLM (latence, tokens, succès)
  - LLMEvaluator    : métriques de qualité sur les résumés et nommages
  - RetryHandler    : retry avec backoff exponentiel + validation JSON
  - ModelRouter     : interface unifiée Ollama / Anthropic Claude

Usage minimal :
    from llmops import ModelRouter, LLMTracer, RetryHandler, evaluator

    router = ModelRouter()
    with LLMTracer("summarize", cluster_id=0) as t:
        response = router.chat(messages=[...])
        t.record(response)
"""

import json
import os
import re
import time
import uuid
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

LLMOPS_LOG_FILE      = os.environ.get("LLMOPS_LOG_FILE",    "logs/llmops_runs.jsonl")
LLMOPS_PROVIDER      = os.environ.get("LLMOPS_PROVIDER",    "ollama")
LLMOPS_MODEL         = os.environ.get("LLMOPS_MODEL",       "llama3")
LLMOPS_MAX_RETRIES   = int(os.environ.get("LLMOPS_MAX_RETRIES",   "3"))
LLMOPS_RETRY_DELAY   = float(os.environ.get("LLMOPS_RETRY_DELAY", "2.0"))

Path(LLMOPS_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("llmops")


# ─────────────────────────────────────────────
# DATACLASS : une ligne de log
# ─────────────────────────────────────────────

@dataclass
class LLMRun:
    run_id:        str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:     str   = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    step:          str   = ""
    cluster_id:    Any   = None
    provider:      str   = LLMOPS_PROVIDER
    model:         str   = LLMOPS_MODEL
    prompt_tokens: int   = 0
    output_tokens: int   = 0
    latency_ms:    float = 0.0
    success:       bool  = False
    retries:       int   = 0
    error:         str   = ""
    quality_score: float = 0.0
    quality_notes: str   = ""

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ─────────────────────────────────────────────
# 1. TRACER
# ─────────────────────────────────────────────

class LLMTracer:
    """
    Context manager qui mesure la latence et sauvegarde un LLMRun dans llmops_runs.jsonl.

    Exemple :
        with LLMTracer("rename", cluster_id=2) as t:
            raw = ollama.chat(...)
            t.record(raw, retries=0)
    """

    def __init__(self, step: str, cluster_id: Any = None):
        self.run = LLMRun(step=step, cluster_id=cluster_id)
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def record(self, response: Any, retries: int = 0, quality_score: float = 0.0, quality_notes: str = ""):
        """Appeler après avoir reçu la réponse du LLM."""
        elapsed = (time.perf_counter() - self._start) * 1000
        self.run.latency_ms    = round(elapsed, 1)
        self.run.retries       = retries
        self.run.success       = True
        self.run.quality_score = round(quality_score, 3)
        self.run.quality_notes = quality_notes

        # Compter les tokens selon le provider
        if isinstance(response, dict):
            usage = response.get("usage")
            if isinstance(usage, dict):
                self.run.prompt_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
                self.run.output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
            else:
                self.run.prompt_tokens = response.get("prompt_eval_count", 0)
                self.run.output_tokens = response.get("eval_count", 0)
        else:
            # Objet Ollama natif
            try:
                self.run.prompt_tokens = getattr(response, "prompt_eval_count", 0) or 0
                self.run.output_tokens = getattr(response, "eval_count", 0) or 0
            except Exception:
                pass

    def fail(self, error: str):
        self.run.error      = str(error)
        self.run.success    = False
        self.run.latency_ms = round((time.perf_counter() - self._start) * 1000, 1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and not self.run.success:
            self.fail(str(exc_val))
        self._flush()
        return False

    def _flush(self):
        try:
            with open(LLMOPS_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(self.run.to_jsonl() + "\n")
        except Exception as e:
            log.warning(f"LLMTracer: impossible d'écrire le log ({e})")

        status = "✓" if self.run.success else "✗"
        log.info(
            f"[{status}] step={self.run.step} cluster={self.run.cluster_id} "
            f"model={self.run.model} latency={self.run.latency_ms}ms "
            f"tokens={self.run.prompt_tokens}+{self.run.output_tokens} "
            f"retries={self.run.retries} quality={self.run.quality_score}"
        )


# ─────────────────────────────────────────────
# 2. EVALUATEUR
# ─────────────────────────────────────────────

class LLMEvaluator:
    """
    Calcule des métriques de qualité légères (sans dépendance lourde).
    """

    MIN_SUMMARY_WORDS = 20
    MAX_SUMMARY_WORDS = 120
    MIN_LABEL_WORDS   = 1
    MAX_LABEL_WORDS   = 5

    def score_summary(self, summary: str, abstract: str = "") -> tuple[float, str]:
        issues = []
        score  = 1.0

        if not summary or summary.strip() in ("", "Résumé non disponible."):
            return 0.0, "résumé vide ou placeholder"

        words = summary.split()
        nw = len(words)

        if nw < self.MIN_SUMMARY_WORDS:
            score -= 0.3
            issues.append(f"trop court ({nw} mots)")
        elif nw > self.MAX_SUMMARY_WORDS:
            score -= 0.15
            issues.append(f"trop long ({nw} mots)")

        fr_markers = {"le", "la", "les", "un", "une", "des", "du", "est", "sont", "dans", "pour", "avec"}
        lower_words = {w.lower().strip(".,;:") for w in words}
        if not lower_words & fr_markers:
            score -= 0.2
            issues.append("probablement pas en français")

        if len(lower_words) / max(nw, 1) < 0.5:
            score -= 0.2
            issues.append("beaucoup de répétitions")

        if abstract:
            ab_words = {w.lower() for w in abstract.split()}
            overlap = len(lower_words & ab_words) / max(len(ab_words), 1)
            if overlap < 0.01:
                score -= 0.1
                issues.append("peu de mots en commun avec l'abstract")

        score = max(0.0, round(score, 2))
        notes = ", ".join(issues) if issues else "OK"
        return score, notes

    def score_cluster_label(self, label: str) -> tuple[float, str]:
        issues = []
        score  = 1.0

        if not label or not label.strip():
            return 0.0, "label vide"

        words = label.strip().split()
        nw = len(words)

        if nw < self.MIN_LABEL_WORDS:
            score -= 0.5
            issues.append("label vide ou un seul caractère")
        elif nw > self.MAX_LABEL_WORDS:
            score -= 0.3
            issues.append(f"label trop long ({nw} mots > {self.MAX_LABEL_WORDS})")

        if re.search(r'["\'\[\]{}]', label):
            score -= 0.2
            issues.append("caractères suspects dans le label")

        score = max(0.0, round(score, 2))
        notes = ", ".join(issues) if issues else "OK"
        return score, notes

    def score_json_output(self, raw: str, expected_len: int) -> tuple[float, str]:
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                return 0.2, "JSON parsé mais pas une liste"
            ratio = len(data) / max(expected_len, 1)
            if ratio < 0.8:
                return 0.5, f"liste trop courte ({len(data)}/{expected_len})"
            return 1.0, "OK"
        except json.JSONDecodeError as e:
            return 0.0, f"JSON invalide: {e}"


evaluator = LLMEvaluator()


# ─────────────────────────────────────────────
# 3. RETRY HANDLER
# ─────────────────────────────────────────────

class RetryHandler:
    """
    Enveloppe un appel LLM avec retry + backoff exponentiel.
    """

    def __init__(self, max_retries: int = LLMOPS_MAX_RETRIES, base_delay: float = LLMOPS_RETRY_DELAY):
        self.max_retries = max_retries
        self.base_delay  = base_delay

    def call(self, fn, validate=None, step: str = ""):
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                result = fn()
                if validate is not None:
                    validated = validate(result)
                    return validated, attempt, ""
                return result, attempt, ""
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    log.warning(f"[retry {attempt+1}/{self.max_retries}] {step} — {e} — attente {delay:.1f}s")
                    time.sleep(delay)
        return None, self.max_retries, last_error


# ─────────────────────────────────────────────
# 4. MODEL ROUTER
# ─────────────────────────────────────────────

class ModelRouter:
    """
    Interface unifiée pour Ollama (local) et Anthropic (cloud).

    FIX : extract_text gère correctement les objets natifs Ollama
          (ChatResponse / Message) en plus des dicts.
    """

    def __init__(self, provider: str = LLMOPS_PROVIDER, model: str = LLMOPS_MODEL):
        self.provider = provider
        self.model    = model

    def chat(self, messages: list[dict], **kwargs) -> Any:
        if self.provider == "anthropic":
            return self._chat_anthropic(messages, **kwargs)
        return self._chat_ollama(messages, **kwargs)

    def extract_text(self, response: Any) -> str:
        """
        Extrait le texte brut d'une réponse, quel que soit le provider.

        Ollama retourne un objet ChatResponse (pas un dict) avec :
            response.message.content  ← texte

        Anthropic retourne un dict normalisé :
            response["content"][0]["text"]
        """
        if self.provider == "anthropic":
            content = response.get("content", [])
            if isinstance(content, list):
                return "".join(c.get("text", "") for c in content if c.get("type") == "text")
            return str(content)

        # ── Ollama : objet natif (ChatResponse) ──────────────────────────────
        # Priorité 1 : attribut .message.content  (ollama-python >= 0.1)
        try:
            msg = response.message
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if content:
                return str(content)
        except AttributeError:
            pass

        # Priorité 2 : dict standard  {"message": {"content": "..."}}
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "")

        # Priorité 3 : conversion str en dernier recours
        return str(response)

    # ── Ollama ──────────────────────────────────────────────────────────────

    def _chat_ollama(self, messages: list[dict], **kwargs) -> Any:
        try:
            import ollama
        except ImportError:
            raise ImportError("ollama non installé : pip install ollama")
        # stream=False obligatoire pour avoir un objet complet (pas un générateur)
        return ollama.chat(model=self.model, messages=messages, stream=False, **kwargs)

    # ── Anthropic ────────────────────────────────────────────────────────────

    def _chat_anthropic(self, messages: list[dict], max_tokens: int = 2048, **kwargs) -> dict:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic non installé : pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("Variable ANTHROPIC_API_KEY non définie")

        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs
        )
        return {
            "content": [{"type": b.type, "text": b.text} for b in msg.content],
            "usage": {
                "input_tokens":  msg.usage.input_tokens,
                "output_tokens": msg.usage.output_tokens,
            },
            "model": msg.model,
        }