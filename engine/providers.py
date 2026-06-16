"""
Lớp truy cập nhà cung cấp model (LLM / Embeddings / Rerank) cho Evaluation Factory.

Tất cả lời gọi mạng đi qua đây để:
  - Tập trung retry + backoff khi gặp lỗi tạm thời (rate-limit 429, timeout).
  - Theo dõi token & chi phí (CostTracker) phục vụ báo cáo "Cost per Eval".
  - Bọc các SDK đồng bộ (groq, openai, cohere) thành coroutine để Runner chạy async.

Cấu hình qua biến môi trường (.env). Nếu thiếu key hoặc provider lỗi, các lớp
gọi đến sẽ tự fallback sang heuristic offline để pipeline luôn chạy được.
"""
from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Bảng giá (USD / 1 triệu token). Dùng để ước lượng chi phí, không phải hoá đơn
# thực. Số liệu lấy theo bảng giá công khai của từng provider tại thời điểm lab.
# --------------------------------------------------------------------------- #
PRICES_PER_1M = {
    "llama-3.3-70b-versatile": {"in": 0.59, "out": 0.79},
    "llama-3.1-8b-instant": {"in": 0.05, "out": 0.08},
    "mistral-large-latest": {"in": 2.00, "out": 6.00},
    "command-r-08-2024": {"in": 0.15, "out": 0.60},
    "openai/gpt-oss-120b": {"in": 0.15, "out": 0.75},
    "gemini-2.5-flash": {"in": 0.30, "out": 2.50},
    # Embeddings / rerank tính theo token / lượt, để riêng bên dưới.
    "embed-multilingual-v3.0": {"in": 0.10, "out": 0.0},
}
RERANK_PRICE_PER_CALL = 2.0 / 1000  # ~$2 / 1000 search (Cohere rerank)

# Đăng ký model logic -> provider cụ thể. Cho phép đổi backend mà không sửa
# code nghiệp vụ (agent / judge chỉ tham chiếu tên logic).
MODELS: Dict[str, Dict[str, str]] = {
    "fast": {"provider": "groq", "model": "llama-3.1-8b-instant", "key_env": "GROQ_API_KEY_2"},
    "strong": {"provider": "groq", "model": "llama-3.3-70b-versatile", "key_env": "GROQ_API_KEY"},
    # Panel giám khảo: 3 HỌ MODEL / NHÀ CUNG CẤP KHÁC NHAU để tránh "thông đồng"
    # (cùng họ model thường có thiên vị tương quan, làm sai lệch consensus).
    "judge_a": {"provider": "groq", "model": "llama-3.3-70b-versatile", "key_env": "GROQ_API_KEY"},      # họ Meta/Llama
    "judge_b": {"provider": "groq", "model": "openai/gpt-oss-120b", "key_env": "GROQ_API_KEY_2"},        # họ OpenAI (chạy trên Groq)
    # Trọng tài khác họ nữa (Google Gemini), chỉ gọi khi A và B bất đồng (cascading).
    # Gemini free ~10 RPM nên chỉ hợp dùng cho số lần gọi ít (tiebreaker).
    # (OpenRouter free đo thực tế ~164s + 429 nên không dùng.)
    "tiebreaker": {"provider": "gemini", "model": "gemini-2.5-flash", "key_env": "GEMINI_API_KEY"},      # họ Google
}

# Giới hạn theo provider để tránh 429 (đo thực tế: Mistral free ~1 req/s).
# concurrency = số request đồng thời tối đa; interval = giãn cách tối thiểu giữa
# 2 request liên tiếp (giây).
_PROVIDER_LIMITS = {
    "groq": {"concurrency": 6, "interval": 0.0},
    "mistral": {"concurrency": 3, "interval": 0.4},
    "openrouter": {"concurrency": 1, "interval": 2.0},
    "cohere": {"concurrency": 4, "interval": 0.0},
    "gemini": {"concurrency": 1, "interval": 4.0},  # free ~10-15 RPM
}

# Giới hạn theo từng MODEL (free tier Groq bị chặn bởi tokens/phút - TPM - cho
# model lớn). Giãn cách interval giúp ở dưới ngưỡng, tránh 429 + backoff lãng phí.
_MODEL_LIMITS = {
    "groq:llama-3.3-70b-versatile": {"concurrency": 3, "interval": 1.1},
    "groq:openai/gpt-oss-120b": {"concurrency": 3, "interval": 1.1},
    "groq:llama-3.1-8b-instant": {"concurrency": 4, "interval": 0.5},
}

EMBED_MODEL = "embed-multilingual-v3.0"
EMBED_DIM = 1024
RERANK_MODEL = "rerank-multilingual-v3.0"

_RETRYABLE_HINTS = ("rate limit", "429", "timeout", "timed out", "overloaded", "503", "502", "temporar")


# --------------------------------------------------------------------------- #
# Theo dõi chi phí
# --------------------------------------------------------------------------- #
@dataclass
class CostTracker:
    """Cộng dồn token & chi phí theo nhãn (agent / judge / embed...)."""

    by_label: Dict[str, Dict[str, float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, label: str, prompt_tokens: int, completion_tokens: int, cost: float, calls: int = 1) -> None:
        with self._lock:
            bucket = self.by_label.setdefault(
                label, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
            )
            bucket["calls"] += calls
            bucket["prompt_tokens"] += prompt_tokens
            bucket["completion_tokens"] += completion_tokens
            bucket["cost_usd"] += cost

    def report(self) -> Dict[str, object]:
        with self._lock:
            total_cost = sum(b["cost_usd"] for b in self.by_label.values())
            total_tokens = sum(b["prompt_tokens"] + b["completion_tokens"] for b in self.by_label.values())
            total_calls = sum(b["calls"] for b in self.by_label.values())
            return {
                "total_cost_usd": round(total_cost, 6),
                "total_tokens": total_tokens,
                "total_calls": total_calls,
                "by_label": {k: {**v, "cost_usd": round(v["cost_usd"], 6)} for k, v in self.by_label.items()},
            }

    def reset(self) -> None:
        with self._lock:
            self.by_label.clear()


# Tracker dùng chung toàn pipeline.
TRACKER = CostTracker()


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = PRICES_PER_1M.get(model)
    if not price:
        return 0.0
    return (prompt_tokens * price["in"] + completion_tokens * price["out"]) / 1_000_000


# --------------------------------------------------------------------------- #
# Client cache (lazy)
# --------------------------------------------------------------------------- #
_clients: Dict[str, object] = {}
_clients_lock = threading.Lock()


def _get_chat_client(provider: str, key_env: str):
    cache_key = f"{provider}:{key_env}"
    with _clients_lock:
        if cache_key in _clients:
            return _clients[cache_key]
        api_key = os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(f"Thiếu API key môi trường {key_env}")
        if provider == "groq":
            from groq import Groq

            client = Groq(api_key=api_key)
        elif provider == "mistral":
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url="https://api.mistral.ai/v1")
        elif provider == "openrouter":
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        elif provider == "gemini":
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        else:
            raise ValueError(f"Provider không hỗ trợ: {provider}")
        _clients[cache_key] = client
        return client


def _get_cohere_client():
    with _clients_lock:
        if "cohere" in _clients:
            return _clients["cohere"]
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError("Thiếu COHERE_API_KEY")
        import cohere

        client = cohere.ClientV2(api_key=api_key)
        _clients["cohere"] = client
        return client


# --------------------------------------------------------------------------- #
# Retry helper (chạy hàm đồng bộ trong thread, có backoff)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Rate limiter theo provider (semaphore concurrency + giãn cách tối thiểu)
# --------------------------------------------------------------------------- #
import time as _time

_sems: Dict[str, asyncio.Semaphore] = {}
_interval_locks: Dict[str, asyncio.Lock] = {}
_last_call: Dict[str, float] = {}


def _cfg(provider: str, model: str = "") -> Dict[str, float]:
    # Ưu tiên cấu hình theo từng model (giới hạn TPM của free tier là theo model),
    # sau đó mới đến cấu hình mặc định của provider.
    key = f"{provider}:{model}"
    if key in _MODEL_LIMITS:
        return _MODEL_LIMITS[key]
    return _PROVIDER_LIMITS.get(provider, {"concurrency": 4, "interval": 0.0})


class _Slot:
    """Context manager async: giữ slot concurrency + giãn cách interval, theo
    từng (provider, model) để tôn trọng giới hạn rate riêng của mỗi model."""

    def __init__(self, provider: str, model: str = ""):
        self.key = f"{provider}:{model}" if model else provider
        self.cfg = _cfg(provider, model)

    async def __aenter__(self):
        sem = _sems.get(self.key)
        if sem is None:
            sem = _sems[self.key] = asyncio.Semaphore(int(self.cfg["concurrency"]))
        self._sem = sem
        await sem.acquire()
        interval = self.cfg["interval"]
        if interval > 0:
            lock = _interval_locks.get(self.key)
            if lock is None:
                lock = _interval_locks[self.key] = asyncio.Lock()
            async with lock:
                wait = interval - (_time.perf_counter() - _last_call.get(self.key, 0.0))
                if wait > 0:
                    await asyncio.sleep(wait)
                _last_call[self.key] = _time.perf_counter()
        return self

    async def __aexit__(self, *exc):
        self._sem.release()
        return False


# Circuit breaker: provider nào bị rate-limit/cạn quota (vd Cohere trial) sẽ bị
# "tạm ngắt" cho phần còn lại của phiên -> các call sau fallback NGAY, không phí
# thời gian retry/backoff vào một dịch vụ đã hết hạn mức.
_disabled_providers: set = set()


def _is_rate_limit(exc: Exception) -> bool:
    m = str(exc).lower()
    return any(h in m for h in ("rate limit", "429", "too many", "toomanyrequests", "quota"))


def _ensure_available(provider: str) -> None:
    if provider in _disabled_providers:
        raise RuntimeError(f"{provider} đã tạm ngắt (rate-limit) trong phiên này")


async def _with_retry(fn, *, retries: int = 4, base_delay: float = 2.0):
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return await asyncio.to_thread(fn)
        except Exception as exc:  # noqa: BLE001 - phân loại bằng nội dung message
            last_exc = exc
            msg = str(exc).lower()
            retryable = any(h in msg for h in _RETRYABLE_HINTS)
            if not retryable or attempt == retries - 1:
                raise
            await asyncio.sleep(base_delay * (2 ** attempt))
    assert last_exc is not None
    raise last_exc


# --------------------------------------------------------------------------- #
# API công khai
# --------------------------------------------------------------------------- #
@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


async def chat(
    role: str,
    messages: List[Dict[str, str]],
    *,
    label: str = "llm",
    temperature: float = 0.0,
    max_tokens: int = 512,
    response_json: bool = False,
) -> ChatResult:
    """Gọi chat completion theo tên model logic (xem MODELS) và ghi nhận chi phí."""
    spec = MODELS[role]
    provider, model = spec["provider"], spec["model"]

    if provider == "cohere":
        text, pt, ct = await _cohere_chat(model, messages, temperature, max_tokens, response_json)
    else:
        client = _get_chat_client(provider, spec["key_env"])

        def _call():
            kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            if response_json:
                kwargs["response_format"] = {"type": "json_object"}
            return client.chat.completions.create(**kwargs)

        async with _Slot(provider, model):
            resp = await _with_retry(_call)
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        text = (resp.choices[0].message.content or "").strip()

    cost = _estimate_cost(model, pt, ct)
    TRACKER.add(label, pt, ct, cost)
    return ChatResult(text=text, model=model, prompt_tokens=pt, completion_tokens=ct, cost_usd=cost)


async def _cohere_chat(model, messages, temperature, max_tokens, response_json):
    """Adapter cho Cohere ClientV2.chat (response shape khác OpenAI)."""
    _ensure_available("cohere")
    client = _get_cohere_client()

    def _call():
        kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if response_json:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat(**kwargs)

    try:
        async with _Slot("cohere"):
            resp = await _with_retry(_call, retries=1)
    except Exception as exc:  # noqa: BLE001
        if _is_rate_limit(exc):
            _disabled_providers.add("cohere")
        raise
    text = (resp.message.content[0].text if resp.message and resp.message.content else "").strip()
    usage = getattr(resp, "usage", None)
    tokens = getattr(usage, "tokens", None) if usage else None
    pt = int(getattr(tokens, "input_tokens", 0) or 0) if tokens else 0
    ct = int(getattr(tokens, "output_tokens", 0) or 0) if tokens else 0
    return text, pt, ct


async def embed(texts: List[str], *, input_type: str = "search_document", label: str = "embed") -> List[List[float]]:
    """Sinh embedding đa ngôn ngữ qua Cohere. input_type: search_document | search_query."""
    _ensure_available("cohere")
    client = _get_cohere_client()

    def _call():
        return client.embed(
            texts=texts,
            model=EMBED_MODEL,
            input_type=input_type,
            embedding_types=["float"],
        )

    try:
        async with _Slot("cohere", EMBED_MODEL):
            resp = await _with_retry(_call, retries=1)  # fail nhanh -> nhường lexical fallback
    except Exception as exc:  # noqa: BLE001
        if _is_rate_limit(exc):
            _disabled_providers.add("cohere")
        raise
    vectors = resp.embeddings.float
    # Cohere tính theo token; ước lượng ~ tổng độ dài / 4 ký tự/token.
    approx_tokens = sum(max(1, len(t) // 4) for t in texts)
    TRACKER.add(label, approx_tokens, 0, _estimate_cost(EMBED_MODEL, approx_tokens, 0))
    return [list(v) for v in vectors]


async def rerank(query: str, documents: List[str], *, top_n: int, label: str = "rerank") -> List[int]:
    """Trả về danh sách index tài liệu đã sắp xếp lại theo độ liên quan (Cohere rerank)."""
    _ensure_available("cohere")
    client = _get_cohere_client()

    def _call():
        return client.rerank(model=RERANK_MODEL, query=query, documents=documents, top_n=top_n)

    try:
        async with _Slot("cohere"):
            resp = await _with_retry(_call, retries=1)
    except Exception as exc:  # noqa: BLE001
        if _is_rate_limit(exc):
            _disabled_providers.add("cohere")
        raise
    TRACKER.add(label, 0, 0, RERANK_PRICE_PER_CALL)
    return [r.index for r in resp.results]
