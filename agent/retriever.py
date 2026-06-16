"""
Retriever: tầng truy xuất (Retrieval) của RAG Agent.

Đường đi chính: Cohere embedding (đa ngôn ngữ) -> Qdrant vector search ->
(tuỳ chọn) Cohere rerank để tăng độ chính xác top-k.

Có fallback từ vựng (lexical / token-overlap) nếu không kết nối được Qdrant
hoặc embedding, đảm bảo toàn pipeline vẫn chạy được khi offline.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional

from data.corpus import load_chunks
from engine import providers

_COLLECTION = "lab14_support_kb"
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class Retriever:
    def __init__(self, collection: str = _COLLECTION, use_rerank: bool = False, candidate_k: int = 10):
        self.collection = collection
        self.use_rerank = use_rerank
        self.candidate_k = candidate_k
        self.chunks = load_chunks()
        self.id_to_chunk = {c["id"]: c for c in self.chunks}
        # index nội bộ <-> chunk id (Qdrant point id là số nguyên)
        self._point_to_chunk = {i: c["id"] for i, c in enumerate(self.chunks)}
        self.mode: str = "uninitialized"  # 'vector' | 'lexical'
        self._client = None
        # Tài nguyên cho fallback lexical
        self._df: Optional[Counter] = None

    # ------------------------------------------------------------------ #
    # Ingest
    # ------------------------------------------------------------------ #
    async def ensure_ingested(self) -> str:
        """Nạp corpus vào Qdrant nếu cần. Trả về mode đã chọn ('vector'/'lexical')."""
        # Luôn dựng sẵn chỉ mục lexical để làm fallback cho TỪNG query khi
        # embedding/Qdrant gặp sự cố (vd Cohere rate-limit giữa chừng).
        self._build_lexical_index()
        try:
            await self._ingest_vector()
            self.mode = "vector"
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Retriever fallback sang lexical (Qdrant/embedding lỗi: {type(exc).__name__}: {str(exc)[:80]})")
            self.mode = "lexical"
        return self.mode

    async def _ingest_vector(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels

        url = os.environ.get("QDRANT_URL")
        api_key = os.environ.get("QDRANT_API_KEY")
        if not url or not api_key:
            raise RuntimeError("Thiếu QDRANT_URL / QDRANT_API_KEY")

        self._client = QdrantClient(url=url, api_key=api_key, timeout=30)

        # Nếu collection đã có đúng số điểm thì bỏ qua (tránh re-embed tốn kém).
        try:
            if self._client.collection_exists(self.collection):
                count = self._client.count(self.collection, exact=True).count
                if count == len(self.chunks):
                    return
        except Exception:  # noqa: BLE001 - lỗi kiểm tra thì cứ tạo lại
            pass

        # Tạo lại collection để đảm bảo nội dung khớp corpus hiện tại.
        self._client.recreate_collection(
            collection_name=self.collection,
            vectors_config=qmodels.VectorParams(size=providers.EMBED_DIM, distance=qmodels.Distance.COSINE),
        )

        vectors = await providers.embed([c["text"] for c in self.chunks], input_type="search_document")
        points = [
            qmodels.PointStruct(
                id=i,
                vector=vectors[i],
                payload={"chunk_id": c["id"], "doc_title": c["doc_title"], "text": c["text"]},
            )
            for i, c in enumerate(self.chunks)
        ]
        self._client.upsert(collection_name=self.collection, points=points, wait=True)

    def _build_lexical_index(self) -> None:
        df: Counter = Counter()
        for c in self.chunks:
            for tok in set(_tokenize(c["text"])):
                df[tok] += 1
        self._df = df

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    async def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.mode == "vector":
            try:
                return await self._search_vector(query, top_k)
            except Exception as exc:  # noqa: BLE001 - embed/Qdrant lỗi giữa chừng -> lexical
                print(f"⚠️ Query embed/search lỗi ({type(exc).__name__}), dùng lexical cho query này.")
                return self._search_lexical(query, top_k)
        return self._search_lexical(query, top_k)

    async def _search_vector(self, query: str, top_k: int) -> List[Dict]:
        from qdrant_client.http import models as qmodels  # noqa: F401 (giữ import cùng client)

        qvec = (await providers.embed([query], input_type="search_query"))[0]
        response = self._client.query_points(
            collection_name=self.collection,
            query=qvec,
            limit=max(self.candidate_k, top_k),
            with_payload=True,
        )
        cand = [
            {"id": p.payload["chunk_id"], "text": p.payload["text"], "score": float(p.score)}
            for p in response.points
        ]
        if self.use_rerank and cand:
            try:
                order = await providers.rerank(query, [c["text"] for c in cand], top_n=min(top_k, len(cand)))
                return [cand[i] for i in order]
            except Exception as exc:  # noqa: BLE001 - rerank lỗi thì dùng thứ tự vector
                print(f"⚠️ Rerank lỗi, dùng thứ tự vector ({type(exc).__name__}).")
        return cand[:top_k]

    def _search_lexical(self, query: str, top_k: int) -> List[Dict]:
        """TF-IDF cosine đơn giản làm fallback offline."""
        assert self._df is not None
        n_docs = len(self.chunks)
        q_tf = Counter(_tokenize(query))

        def idf(tok: str) -> float:
            return math.log((n_docs + 1) / (self._df.get(tok, 0) + 1)) + 1.0

        def vec(tf: Counter) -> Dict[str, float]:
            return {tok: freq * idf(tok) for tok, freq in tf.items()}

        qv = vec(q_tf)
        q_norm = math.sqrt(sum(v * v for v in qv.values())) or 1.0

        scored = []
        for c in self.chunks:
            dv = vec(Counter(_tokenize(c["text"])))
            dot = sum(qv.get(t, 0.0) * dv.get(t, 0.0) for t in qv)
            d_norm = math.sqrt(sum(v * v for v in dv.values())) or 1.0
            scored.append({"id": c["id"], "text": c["text"], "score": dot / (q_norm * d_norm)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
