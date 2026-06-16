"""
Đánh giá tầng Retrieval (Hit Rate, MRR) và các chỉ số RAGAS-style
(faithfulness, answer relevancy) cho từng câu trả lời của Agent.

Faithfulness / relevancy ở đây dùng phép đo cosine trên embedding (proxy nhẹ,
không gọi thêm LLM) — đủ để so sánh tương đối giữa V1/V2 và phân cụm lỗi:
  - relevancy   = cosine(answer, question): câu trả lời có bám sát câu hỏi không.
  - faithfulness = max cosine(answer, context_i): câu trả lời có "tựa" vào ngữ
                   cảnh đã truy xuất không (chống bịa - hallucination).

Tính đúng/sai về mặt nội dung so với Ground Truth do Multi-Judge đảm nhiệm.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

from engine import providers

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _token_overlap(a: str, b: str) -> float:
    ta, tb = set(_TOKEN_RE.findall(a.lower())), set(_TOKEN_RE.findall(b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class RetrievalEvaluator:
    """Các chỉ số truy xuất thuần tuý (không phụ thuộc mạng)."""

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """1.0 nếu có ít nhất 1 expected_id nằm trong top_k của retrieved_ids."""
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """Mean Reciprocal Rank: 1/vị_trí (1-indexed) của expected_id đầu tiên gặp được."""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    def evaluate_batch(self, results: List[Dict], top_k: int = 3) -> Dict:
        """
        Tổng hợp Hit Rate / MRR trên các kết quả benchmark.
        Chỉ tính trên các case CÓ ground truth retrieval (expected_retrieval_ids khác rỗng);
        các case out-of-context (không có chunk đúng) được bỏ qua khỏi mẫu số.
        """
        hits, mrrs = [], []
        for r in results:
            expected = r.get("expected_retrieval_ids") or []
            retrieved = r.get("retrieved_ids") or []
            if not expected:
                continue
            hits.append(self.calculate_hit_rate(expected, retrieved, top_k))
            mrrs.append(self.calculate_mrr(expected, retrieved))
        n = len(hits)
        return {
            "avg_hit_rate": sum(hits) / n if n else 0.0,
            "avg_mrr": sum(mrrs) / n if n else 0.0,
            "scored_cases": n,
        }


class RagEvaluator:
    """
    Evaluator dùng cho BenchmarkRunner: trả về faithfulness, relevancy và chỉ số
    retrieval (hit_rate, mrr) cho từng case. Interface: async score(case, resp).
    """

    def __init__(self, top_k: int = 3):
        self.retrieval = RetrievalEvaluator()
        self.top_k = top_k

    async def score(self, case: Dict, resp: Dict) -> Dict:
        expected_ids = case.get("expected_retrieval_ids") or []
        retrieved_ids = resp.get("retrieved_ids") or []
        contexts = resp.get("contexts") or []
        answer = resp.get("answer") or ""
        question = case.get("question") or ""

        # --- Retrieval (chỉ áp dụng khi có ground truth chunk) ---
        if expected_ids:
            retrieval = {
                "hit_rate": self.retrieval.calculate_hit_rate(expected_ids, retrieved_ids, self.top_k),
                "mrr": self.retrieval.calculate_mrr(expected_ids, retrieved_ids),
            }
        else:
            retrieval = None  # out-of-context: không tính vào mẫu số

        # --- Faithfulness / Relevancy qua embedding (fallback token-overlap) ---
        try:
            vectors = await providers.embed([answer, question] + contexts, input_type="search_query", label="eval_embed")
            ans_vec, q_vec = vectors[0], vectors[1]
            ctx_vecs = vectors[2:]
            relevancy = max(0.0, _cosine(ans_vec, q_vec))
            faithfulness = max((max(0.0, _cosine(ans_vec, cv)) for cv in ctx_vecs), default=0.0)
        except Exception:  # noqa: BLE001 - offline / lỗi mạng -> heuristic
            relevancy = _token_overlap(answer, question)
            faithfulness = max((_token_overlap(answer, c) for c in contexts), default=0.0)

        return {
            "faithfulness": round(faithfulness, 4),
            "relevancy": round(relevancy, 4),
            "retrieval": retrieval,
        }
