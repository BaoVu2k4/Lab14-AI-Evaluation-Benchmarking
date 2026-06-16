"""
BenchmarkRunner: chạy toàn bộ test case qua Agent -> Evaluator -> Multi-Judge
một cách BẤT ĐỒNG BỘ (async) theo batch để vừa nhanh vừa tránh rate-limit.

Mỗi case được bọc try/except: một case lỗi sẽ được đánh dấu 'error' thay vì làm
sập toàn bộ benchmark. Runner cũng đo latency và gom token/cost cho từng case.
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict, List


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()
        try:
            # 1. Gọi Agent (Retrieval + Generation)
            response = await self.agent.query(test_case["question"])
            latency = time.perf_counter() - start_time

            # 2. RAGAS-style + Retrieval metrics
            ragas_scores = await self.evaluator.score(test_case, response)

            # 3. Multi-Judge consensus
            judge_result = await self.judge.evaluate_multi_judge(
                test_case["question"],
                response["answer"],
                test_case["expected_answer"],
            )

            final_score = judge_result["final_score"]
            return {
                "id": test_case.get("id"),
                "test_case": test_case["question"],
                "case_type": test_case.get("metadata", {}).get("type"),
                "difficulty": test_case.get("metadata", {}).get("difficulty"),
                "agent_response": response["answer"],
                "expected_answer": test_case["expected_answer"],
                "expected_retrieval_ids": test_case.get("expected_retrieval_ids", []),
                "retrieved_ids": response.get("retrieved_ids", []),
                "latency": round(latency, 3),
                "tokens": response.get("metadata", {}).get("tokens_used", 0),
                "cost_usd": response.get("metadata", {}).get("cost_usd", 0.0),
                "ragas": ragas_scores,
                "judge": judge_result,
                "status": "fail" if final_score < 3 else "pass",
            }
        except Exception as exc:  # noqa: BLE001 - cô lập lỗi từng case
            return {
                "id": test_case.get("id"),
                "test_case": test_case.get("question", ""),
                "agent_response": f"[ERROR: {type(exc).__name__}: {str(exc)[:100]}]",
                "expected_answer": test_case.get("expected_answer", ""),
                "expected_retrieval_ids": test_case.get("expected_retrieval_ids", []),
                "retrieved_ids": [],
                "latency": round(time.perf_counter() - start_time, 3),
                "tokens": 0,
                "cost_usd": 0.0,
                "ragas": {"faithfulness": 0.0, "relevancy": 0.0, "retrieval": None},
                "judge": {"final_score": 0, "agreement_rate": 0.0, "conflict": False,
                          "individual_scores": {}, "reasoning": "error"},
                "status": "error",
            }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        """Chạy song song theo batch_size để cân bằng tốc độ và rate-limit."""
        results: List[Dict] = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            print(f"   ...đã chạy {min(i + batch_size, len(dataset))}/{len(dataset)} case")
        return results
