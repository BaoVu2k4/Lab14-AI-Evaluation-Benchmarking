"""
RAG Agent đánh giá. Có 2 phiên bản để phục vụ Regression Testing:

  - V1 (base):     model nhỏ (8B), KHÔNG rerank, prompt lỏng -> dễ bịa khi
                   ngữ cảnh không chứa câu trả lời (hallucination).
  - V2 (optimized): model mạnh (70B) + Cohere rerank + prompt grounded chặt
                   (bắt buộc trả lời chỉ dựa trên ngữ cảnh, nếu thiếu thì nói
                   "không tìm thấy thông tin"). Mục tiêu giảm hallucination,
                   tăng faithfulness và hit rate top-k.

Mỗi lần query trả về retrieved_ids để engine tính Hit Rate / MRR.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from agent.retriever import Retriever
from engine import providers

_PROMPT_V1 = (
    "Bạn là trợ lý hỗ trợ khách hàng của sàn TMĐT ShopVN. Hãy trả lời câu hỏi của "
    "khách một cách hữu ích và thân thiện dựa trên thông tin tham khảo dưới đây.\n\n"
    "Thông tin tham khảo:\n{context}\n\nCâu hỏi: {question}\nTrả lời:"
)

_PROMPT_V2 = (
    "Bạn là trợ lý hỗ trợ khách hàng của sàn TMĐT ShopVN. Tuân thủ nghiêm các quy tắc sau:\n"
    "1. CHỈ trả lời dựa trên 'Ngữ cảnh' được cung cấp. Tuyệt đối không bịa thông tin.\n"
    "2. Nếu Ngữ cảnh không chứa thông tin để trả lời, hãy nói rõ: 'Xin lỗi, tôi không tìm "
    "thấy thông tin này trong tài liệu hỗ trợ. Bạn vui lòng liên hệ tổng đài 1900-1234.'\n"
    "3. Trả lời ngắn gọn, chính xác, đúng trọng tâm câu hỏi.\n\n"
    "Ngữ cảnh:\n{context}\n\nCâu hỏi: {question}\nTrả lời:"
)


class MainAgent:
    """Agent RAG có cấu hình theo phiên bản."""

    def __init__(self, version: str = "v1", retriever: Optional[Retriever] = None, top_k: int = 3):
        self.version = version
        self.top_k = top_k
        if version == "v2":
            self.model_role = "strong"
            self.prompt_tmpl = _PROMPT_V2
            self.retriever = retriever or Retriever(use_rerank=True)
        else:
            self.model_role = "fast"
            self.prompt_tmpl = _PROMPT_V1
            self.retriever = retriever or Retriever(use_rerank=False)
        self.name = f"ShopVN-SupportAgent-{version}"

    async def setup(self) -> None:
        await self.retriever.ensure_ingested()

    async def query(self, question: str) -> Dict:
        hits = await self.retriever.search(question, top_k=self.top_k)
        contexts = [h["text"] for h in hits]
        retrieved_ids = [h["id"] for h in hits]

        context_block = "\n".join(f"- {c}" for c in contexts) if contexts else "(không có)"
        prompt = self.prompt_tmpl.format(context=context_block, question=question)

        try:
            result = await providers.chat(
                self.model_role,
                [{"role": "user", "content": prompt}],
                label=f"agent_{self.version}",
                temperature=0.0,
                max_tokens=250,
            )
            answer = result.text
            tokens = result.prompt_tokens + result.completion_tokens
            cost = result.cost_usd
            model = result.model
        except Exception as exc:  # noqa: BLE001 - không để 1 case làm sập cả benchmark
            answer = f"[LỖI SINH CÂU TRẢ LỜI: {type(exc).__name__}]"
            tokens, cost, model = 0, 0.0, self.model_role

        return {
            "answer": answer,
            "contexts": contexts,
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "version": self.version,
                "model": model,
                "tokens_used": tokens,
                "cost_usd": cost,
                "reranked": self.retriever.use_rerank,
            },
        }


if __name__ == "__main__":
    import asyncio

    async def _demo():
        agent = MainAgent("v2")
        await agent.setup()
        resp = await agent.query("Làm thế nào để đổi mật khẩu?")
        print(resp["answer"])
        print("retrieved:", resp["retrieved_ids"], "| tokens:", resp["metadata"]["tokens_used"])

    asyncio.run(_demo())
