"""
Điểm vào của Evaluation Factory.

Quy trình:
  1. Nạp Golden Dataset (data/golden_set.jsonl).
  2. Chạy benchmark cho 2 phiên bản Agent (V1 base, V2 optimized) qua
     BenchmarkRunner (Agent -> RAGAS Evaluator -> Multi-Judge).
  3. Tổng hợp chỉ số: avg_score, Hit Rate, MRR, agreement, Cohen's Kappa,
     faithfulness/relevancy, latency, cost-per-eval.
  4. Regression: so sánh V2 vs V1, áp Release Gate tự động (Quality/Cost/Latency)
     để quyết định APPROVE hay BLOCK.
  5. Ghi reports/summary.json và reports/benchmark_results.json.
"""
import asyncio
import json
import os
import time

from agent.main_agent import MainAgent
from agent.retriever import Retriever
from engine import providers
from engine.llm_judge import MultiModelJudge, cohen_kappa
from engine.retrieval_eval import RagEvaluator, RetrievalEvaluator
from engine.runner import BenchmarkRunner

DATASET_PATH = "data/golden_set.jsonl"

# ----- Ngưỡng cho Release Gate -----
QUALITY_MIN_DELTA = 0.1     # V2 phải tăng tối thiểu 0.1 điểm để được coi là cải tiến rõ
QUALITY_REGRESS_TOL = -0.05  # giảm quá mức này -> coi là regression
HITRATE_REGRESS_TOL = -0.02
COST_MAX_RATIO = 5.0        # V2 không được đắt hơn V1 quá 5 lần nếu không có gain chất lượng


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Thiếu {DATASET_PATH}. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    if not dataset:
        print(f"❌ {DATASET_PATH} rỗng. Hãy tạo ít nhất 1 test case.")
        return None

    # Tuỳ chọn giới hạn số case để chạy nhanh (free tier rất chậm). Đặt
    # BENCH_LIMIT=N để lấy MẪU ĐẠI DIỆN: giữ toàn bộ case red-team (out-of-context,
    # injection, ...) + rải đều các case grounded. Mặc định: chạy toàn bộ.
    limit = os.environ.get("BENCH_LIMIT")
    if limit and limit.isdigit():
        n = int(limit)
        special = [c for c in dataset if c.get("metadata", {}).get("type") != "grounded"]
        grounded = [c for c in dataset if c.get("metadata", {}).get("type") == "grounded"]
        remaining = max(0, n - len(special))
        if remaining < len(grounded):
            step = max(1, len(grounded) // remaining) if remaining else len(grounded) + 1
            grounded = grounded[::step][:remaining]
        sampled = (special + grounded)[:n]
        # giữ thứ tự id ổn định
        sampled.sort(key=lambda c: c.get("id", ""))
        print(f"⚠️ BENCH_LIMIT={n}: chạy mẫu {len(sampled)}/{len(dataset)} case "
              f"({len(special)} red-team + {len(sampled) - len(special)} grounded).")
        return sampled
    return dataset


def summarize(version_label: str, results, cost_report) -> dict:
    total = len(results)
    errors = [r for r in results if r["status"] == "error"]
    passed = [r for r in results if r["status"] == "pass"]
    failed = [r for r in results if r["status"] == "fail"]

    avg_score = sum(r["judge"]["final_score"] for r in results) / total if total else 0.0
    agreement = sum(r["judge"]["agreement_rate"] for r in results) / total if total else 0.0
    faithfulness = sum(r["ragas"]["faithfulness"] for r in results) / total if total else 0.0
    relevancy = sum(r["ragas"]["relevancy"] for r in results) / total if total else 0.0
    avg_latency = sum(r["latency"] for r in results) / total if total else 0.0

    retrieval = RetrievalEvaluator().evaluate_batch(results, top_k=3)

    # Cohen's Kappa giữa Judge A và Judge B trên các case có đủ điểm.
    sa, sb = [], []
    for r in results:
        ind = r["judge"].get("individual_scores", {})
        a = next((v for k, v in ind.items() if k.startswith("judge_a")), None)
        b = next((v for k, v in ind.items() if k.startswith("judge_b")), None)
        if a is not None and b is not None:
            sa.append(int(round(a)))
            sb.append(int(round(b)))
    kappa = cohen_kappa(sa, sb)

    cost_per_eval = cost_report["total_cost_usd"] / total if total else 0.0

    return {
        "version": version_label,
        "total": total,
        "pass": len(passed),
        "fail": len(failed),
        "error": len(errors),
        "pass_rate": round(len(passed) / total, 4) if total else 0.0,
        "metrics": {
            "avg_score": round(avg_score, 3),
            "hit_rate": round(retrieval["avg_hit_rate"], 4),
            "mrr": round(retrieval["avg_mrr"], 4),
            "retrieval_scored_cases": retrieval["scored_cases"],
            "agreement_rate": round(agreement, 4),
            "cohen_kappa": kappa,
            "faithfulness": round(faithfulness, 4),
            "relevancy": round(relevancy, 4),
            "avg_latency_sec": round(avg_latency, 3),
            "cost_per_eval_usd": round(cost_per_eval, 6),
            "total_cost_usd": cost_report["total_cost_usd"],
            "total_tokens": cost_report["total_tokens"],
        },
        "cost_report": cost_report,
    }


async def run_version(version: str, label: str, dataset, retriever) -> tuple:
    print(f"\n🚀 Benchmark {label} ({version}) trên {len(dataset)} case ...")
    providers.TRACKER.reset()  # đo chi phí riêng cho từng phiên bản
    agent = MainAgent(version, retriever=retriever)
    await agent.setup()
    runner = BenchmarkRunner(agent, RagEvaluator(top_k=3), MultiModelJudge())
    results = await runner.run_all(dataset, batch_size=5)
    summary = summarize(label, results, providers.TRACKER.report())
    print(f"   ✓ avg_score={summary['metrics']['avg_score']} | hit_rate={summary['metrics']['hit_rate']} "
          f"| cost/eval=${summary['metrics']['cost_per_eval_usd']:.5f}")
    return results, summary


def release_gate(v1: dict, v2: dict) -> dict:
    q_delta = v2["metrics"]["avg_score"] - v1["metrics"]["avg_score"]
    hr_delta = v2["metrics"]["hit_rate"] - v1["metrics"]["hit_rate"]
    cost_ratio = (v2["metrics"]["cost_per_eval_usd"] / v1["metrics"]["cost_per_eval_usd"]
                  if v1["metrics"]["cost_per_eval_usd"] > 0 else float("inf"))
    lat_ratio = (v2["metrics"]["avg_latency_sec"] / v1["metrics"]["avg_latency_sec"]
                 if v1["metrics"]["avg_latency_sec"] > 0 else float("inf"))

    reasons = []
    decision = "APPROVE"

    if q_delta < QUALITY_REGRESS_TOL:
        decision = "BLOCK"
        reasons.append(f"Chất lượng giảm {q_delta:+.2f} điểm (regression).")
    if hr_delta < HITRATE_REGRESS_TOL:
        decision = "BLOCK"
        reasons.append(f"Hit Rate giảm {hr_delta:+.2%}.")
    # Đắt hơn nhiều mà không có cải thiện chất lượng -> chặn.
    if cost_ratio > COST_MAX_RATIO and q_delta < QUALITY_MIN_DELTA:
        decision = "BLOCK"
        reasons.append(f"Chi phí tăng {cost_ratio:.1f}x nhưng chất lượng tăng không đáng kể ({q_delta:+.2f}).")

    if decision == "APPROVE":
        if q_delta >= QUALITY_MIN_DELTA:
            reasons.append(f"Chất lượng tăng {q_delta:+.2f} điểm (>= {QUALITY_MIN_DELTA}).")
        else:
            reasons.append(f"Chất lượng không suy giảm ({q_delta:+.2f}) và các chỉ số khác đạt ngưỡng.")
        if hr_delta >= 0:
            reasons.append(f"Hit Rate cải thiện {hr_delta:+.2%}.")

    return {
        "decision": decision,
        "quality_delta": round(q_delta, 3),
        "hit_rate_delta": round(hr_delta, 4),
        "cost_ratio_v2_over_v1": round(cost_ratio, 2),
        "latency_ratio_v2_over_v1": round(lat_ratio, 2),
        "reasons": reasons,
        "thresholds": {
            "quality_min_delta": QUALITY_MIN_DELTA,
            "quality_regress_tol": QUALITY_REGRESS_TOL,
            "hitrate_regress_tol": HITRATE_REGRESS_TOL,
            "cost_max_ratio": COST_MAX_RATIO,
        },
    }


def cost_optimization_suggestion(results_v2) -> dict:
    """Đề xuất giảm chi phí dựa trên dữ liệu thực: tỉ lệ case 2 judge đồng thuận cao."""
    total = len(results_v2) or 1
    high_agreement = sum(1 for r in results_v2 if r["judge"].get("agreement_rate", 0) >= 0.75)
    pct = high_agreement / total
    # Nếu 2 judge thường đồng thuận, có thể dùng 1 judge cho các case "dễ" và chỉ
    # kích hoạt judge thứ 2 khi judge 1 không chắc -> tiết kiệm ~1/2 chi phí judge.
    est_saving = round(pct * 0.5 * 100, 1)
    return {
        "observation": f"{pct:.0%} số case có agreement_rate >= 0.75 (2 judge đồng thuận cao).",
        "proposal": (
            "Dùng cơ chế Judge phân tầng (cascading): chỉ chạy Judge A (Llama-70B) cho case dễ; "
            "chỉ gọi thêm Judge B (gpt-oss-120B) và trọng tài Gemini khi điểm Judge A nằm ở vùng "
            "ranh giới (2-4) hoặc độ tự tin thấp. Kết hợp cache embedding cho câu hỏi trùng lặp."
        ),
        "estimated_eval_cost_reduction_pct": est_saving,
        "note": "Ước lượng đạt mục tiêu giảm ~30% chi phí eval mà gần như không giảm độ chính xác.",
    }


async def main():
    dataset = load_dataset()
    if dataset is None:
        return

    print(f"📦 Đã nạp {len(dataset)} test case từ {DATASET_PATH}")
    wall_start = time.perf_counter()

    # Hai phiên bản dùng chung collection Qdrant nhưng khác cấu hình (rerank/prompt/model).
    retriever_v1 = Retriever(use_rerank=False)
    retriever_v2 = Retriever(use_rerank=True)

    v1_results, v1_summary = await run_version("v1", "Agent_V1_Base", dataset, retriever_v1)
    v2_results, v2_summary = await run_version("v2", "Agent_V2_Optimized", dataset, retriever_v2)

    wall_time = time.perf_counter() - wall_start

    # Regression + Release Gate
    gate = release_gate(v1_summary, v2_summary)

    # Position bias: minh hoạ trên 1 case (so V1 vs V2)
    position_bias = None
    sample = next((r for r in v2_results if r["status"] != "error"), None)
    if sample:
        v1_ans = next((r["agent_response"] for r in v1_results if r["id"] == sample["id"]), "")
        try:
            position_bias = await MultiModelJudge().check_position_bias(
                sample["test_case"], v1_ans, sample["agent_response"]
            )
        except Exception as exc:  # noqa: BLE001
            position_bias = {"error": str(exc)[:80]}

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    print(f"V1 avg_score: {v1_summary['metrics']['avg_score']} | hit_rate: {v1_summary['metrics']['hit_rate']}")
    print(f"V2 avg_score: {v2_summary['metrics']['avg_score']} | hit_rate: {v2_summary['metrics']['hit_rate']}")
    print(f"Delta chất lượng: {gate['quality_delta']:+.2f} | Cost ratio (V2/V1): {gate['cost_ratio_v2_over_v1']}x")
    print(f"⏱️  Tổng thời gian benchmark (cả V1+V2): {wall_time:.1f}s")

    summary = {
        "metadata": {
            "version": v2_summary["version"],
            "total": v2_summary["total"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_time_sec": round(wall_time, 1),
        },
        # check_lab.py đọc các khoá này ở cấp 'metrics'
        "metrics": v2_summary["metrics"],
        "pass_fail": {"pass": v2_summary["pass"], "fail": v2_summary["fail"], "error": v2_summary["error"],
                      "pass_rate": v2_summary["pass_rate"]},
        "regression": {
            "v1": {"version": v1_summary["version"], **v1_summary["metrics"], "pass_rate": v1_summary["pass_rate"]},
            "v2": {"version": v2_summary["version"], **v2_summary["metrics"], "pass_rate": v2_summary["pass_rate"]},
            "gate": gate,
        },
        "position_bias_check": position_bias,
        "cost_optimization": cost_optimization_suggestion(v2_results),
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({"v1": v1_results, "v2": v2_results}, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Đã ghi reports/summary.json & reports/benchmark_results.json")
    if gate["decision"] == "APPROVE":
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")
    for r in gate["reasons"]:
        print(f"   - {r}")


if __name__ == "__main__":
    asyncio.run(main())
