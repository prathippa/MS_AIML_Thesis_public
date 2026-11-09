"""
generation.py
-------------
Stage 2 (baseline generation). Query Model A (Claude Sonnet 5.5, Anthropic) and
Model B (Llama 3.3 70B, Groq) with ONE identical prompt and collect a fixed
structured output per case:

    {hts_code: 10 digits, confidence: 0..1, rationale: str, evidence_ids: [str]}

Design-lock guarantees implemented here:
  * identical prompt across models (only the client differs);
  * malformed output is rejected and re-prompted ONCE under the same settings;
  * every response is cached and timestamped, so (a) a later API change never
    alters the experimental dataset and (b) re-running analysis never re-bills;
  * runs are resumable -- a case already in the cache is skipped;
  * a hard cap on API calls prevents a bug from running up charges.

The real Anthropic/Groq SDKs are imported lazily, so this module imports and is
fully testable with a MockClient even when the SDKs are not installed.
"""
from __future__ import annotations
import json, os, re, time, hashlib, datetime
from dataclasses import dataclass, field, asdict
from typing import Protocol

import sys
sys.path.insert(0, os.path.dirname(__file__))
from hts_hierarchy import normalize_hts  # noqa: E402


# --------------------------------------------------------------------------- #
# Prompt (identical across models)                                            #
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a customs tariff classification assistant for the United States "
    "Harmonized Tariff Schedule (HTS). Given a product description and supporting "
    "tariff evidence, assign the single most appropriate 10-digit U.S. HTS code.\n\n"
    "Return ONLY a JSON object, with no surrounding text, in exactly this form:\n"
    '{"hts_code": "<10 digits>", "confidence": <number between 0 and 1>, '
    '"rationale": "<one or two sentences>", "evidence_ids": ["<id>", ...]}\n\n'
    "Rules:\n"
    "- hts_code must be exactly 10 digits (keep leading zeros).\n"
    "- confidence is your estimated probability that the FULL 10-digit code is "
    "correct, as a decimal from 0 to 1.\n"
    "- evidence_ids lists the identifiers of the supplied evidence items you relied on.\n"
    "- Output JSON only."
)


def build_messages(case: dict, evidence: list[dict]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt). `case` needs 'description'; each
    evidence item needs 'id' and 'text'. Identical for every model."""
    ev_lines = "\n".join(f"[{e['id']}] {e['text']}" for e in evidence) or "(no evidence provided)"
    user = (
        f"Product description:\n{case['description']}\n\n"
        f"Tariff evidence:\n{ev_lines}\n\n"
        "Classify this product. Respond with the JSON object only."
    )
    return SYSTEM_PROMPT, user


# --------------------------------------------------------------------------- #
# Parsing & validation                                                        #
# --------------------------------------------------------------------------- #
class MalformedOutput(ValueError):
    pass


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply, tolerating ``` fences
    and stray prose around it."""
    if text is None:
        raise MalformedOutput("empty response")
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # fall back to the first balanced {...} block, ignoring braces inside strings
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        return json.loads(t[start:i + 1])
                    except json.JSONDecodeError:
                        start = None
    raise MalformedOutput("no JSON object found")


def _coerce_confidence(v) -> float:
    """Confidence must be a decimal in [0, 1]. An explicit percent string
    ("85%") is honoured; a bare out-of-range number ("85", "2") is treated as
    malformed and retried, rather than silently rescaled -- silent rescaling of
    an ambiguous value could corrupt the calibration data this study depends on.
    """
    if isinstance(v, str):
        s = v.strip()
        try:
            c = float(s[:-1]) / 100.0 if s.endswith("%") else float(s)
        except ValueError:
            raise MalformedOutput(f"confidence not a number: {v!r}")
    else:
        try:
            c = float(v)
        except (TypeError, ValueError):
            raise MalformedOutput(f"confidence not a number: {v!r}")
    if not (0.0 <= c <= 1.0):
        raise MalformedOutput(f"confidence out of range: {c}")
    return c


def parse_response(text: str) -> dict:
    """Validate a raw reply into the fixed schema or raise MalformedOutput."""
    obj = _extract_json(text)
    if not isinstance(obj, dict):
        raise MalformedOutput("top-level JSON is not an object")
    raw_code = obj.get("hts_code")
    digits = re.sub(r"\D", "", str(raw_code)) if raw_code is not None else ""
    if len(digits) != 10:
        raise MalformedOutput(f"hts_code is not 10 digits: {raw_code!r}")
    conf = _coerce_confidence(obj.get("confidence"))
    rationale = obj.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise MalformedOutput("rationale missing or empty")
    ev = obj.get("evidence_ids", [])
    if not isinstance(ev, list):
        ev = [str(ev)]
    return {
        "hts_code": normalize_hts(digits),
        "confidence": conf,
        "rationale": rationale.strip(),
        "evidence_ids": [str(x) for x in ev],
    }


# --------------------------------------------------------------------------- #
# Clients (lazy real SDKs; mock for testing)                                  #
# --------------------------------------------------------------------------- #
class LLMClient(Protocol):
    name: str
    def complete(self, system: str, user: str, decoding: dict) -> str: ...


class AnthropicClient:
    """Model A -- Claude Sonnet 5.5."""
    def __init__(self, model: str, api_key: str):
        import anthropic  # lazy
        self.name, self.model = model, model
        self._c = anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, user: str, decoding: dict) -> str:
        msg = self._c.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=decoding["temperature"],
            top_p=decoding["top_p"],
            max_tokens=decoding["max_tokens"],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


class GroqClient:
    """Model B -- Llama 3.3 70B (OpenAI-compatible chat API)."""
    def __init__(self, model: str, api_key: str):
        from groq import Groq  # lazy
        self.name, self.model = model, model
        self._c = Groq(api_key=api_key)

    def complete(self, system: str, user: str, decoding: dict) -> str:
        r = self._c.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=decoding["temperature"],
            top_p=decoding["top_p"],
            max_tokens=decoding["max_tokens"],
            seed=decoding.get("seed"),
        )
        return r.choices[0].message.content


class MockClient:
    """Returns queued canned replies; used for offline tests."""
    def __init__(self, model: str, replies: list[str]):
        self.name, self.model, self._replies, self.calls = model, model, list(replies), 0
    def complete(self, system: str, user: str, decoding: dict) -> str:
        r = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return r


def get_client(model_cfg: dict) -> LLMClient:
    key = os.environ.get(model_cfg["api_key_env"])
    if not key:
        raise RuntimeError(f"Missing API key env var {model_cfg['api_key_env']}")
    if model_cfg["provider"] == "anthropic":
        return AnthropicClient(model_cfg["model"], key)
    if model_cfg["provider"] == "groq":
        return GroqClient(model_cfg["model"], key)
    raise ValueError(f"Unknown provider {model_cfg['provider']!r}")


# --------------------------------------------------------------------------- #
# Cache (append-only JSONL, resume-safe)                                       #
# --------------------------------------------------------------------------- #
class ResponseCache:
    def __init__(self, path: str):
        self.path = path
        self.done: set[str] = set()
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        self.done.add(json.loads(line)["case_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    def has(self, case_id: str) -> bool:
        return case_id in self.done
    def append(self, record: dict) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        self.done.add(record["case_id"])


# --------------------------------------------------------------------------- #
# Generation                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class GenRecord:
    case_id: str
    model: str
    status: str                       # "ok" | "failed"
    timestamp: str
    decoding: dict
    prompt_hash: str
    attempts: int
    hts_code: str | None = None
    confidence: float | None = None
    rationale: str | None = None
    evidence_ids: list = field(default_factory=list)
    raw: str | None = None
    error: str | None = None


def _prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256((system + "\x00" + user).encode()).hexdigest()[:16]


def generate_one(client, case, evidence, decoding, retry_once=True) -> GenRecord:
    system, user = build_messages(case, evidence)
    ph = _prompt_hash(system, user)
    attempts, last_err, raw = 0, None, None
    for attempt in range(2 if retry_once else 1):
        attempts += 1
        raw = client.complete(system, user, decoding)
        try:
            parsed = parse_response(raw)
            return GenRecord(
                case_id=case["id"], model=client.model, status="ok",
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                decoding=decoding, prompt_hash=ph, attempts=attempts, raw=raw, **parsed,
            )
        except MalformedOutput as e:
            last_err = str(e)
    return GenRecord(
        case_id=case["id"], model=client.model, status="failed",
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        decoding=decoding, prompt_hash=ph, attempts=attempts, raw=raw, error=last_err,
    )


def run_generation(client, cases, evidence_by_id, decoding, cache_path,
                   max_calls=2000, retry_once=True, verbose=False):
    """Iterate cases, skipping cached ones; enforce a hard call cap. Returns records."""
    cache = ResponseCache(cache_path)
    calls, out = 0, []
    for case in cases:
        if cache.has(case["id"]):
            continue
        budget = 2 if retry_once else 1
        if calls + budget > max_calls:
            if verbose:
                print(f"Call cap {max_calls} reached; stopping at {case['id']}.")
            break
        rec = generate_one(client, case, evidence_by_id.get(case["id"], []), decoding, retry_once)
        calls += rec.attempts
        cache.append(asdict(rec))
        out.append(rec)
        if verbose:
            print(f"{case['id']}: {rec.status} ({rec.hts_code}, conf={rec.confidence})")
    return out
