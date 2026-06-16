"""
Multi-Judge Consensus Engine.

Dùng panel giám khảo 3 HỌ MODEL KHÁC NHAU (tránh "thông đồng" do cùng họ), cơ
chế "cascading":
  - Judge A: Groq llama-3.3-70b-versatile (họ Meta/Llama)
  - Judge B: Groq openai/gpt-oss-120b (họ OpenAI - chạy trên hạ tầng Groq)
  - Tiebreaker (trọng tài): Google gemini-2.5-flash (họ Google) -> chỉ gọi khi
    A và B bất đồng. Vừa cho góc nhìn chéo họ model khi cần, vừa tiết kiệm chi
    phí (đa số case không cần judge thứ 3).

Cơ chế:
  - Mỗi judge chấm 1-5 trên các tiêu chí (accuracy, completeness, tone, safety)
    và đưa ra điểm tổng (overall) kèm lý do.
  - Agreement rate = 1 - |A - B| / 4 (chuẩn hoá về [0,1]).
  - Xử lý xung đột: nếu |A - B| > 1 -> gọi Tiebreaker (Mistral) và lấy
    TRUNG VỊ của 3 điểm làm final_score; ngược lại lấy trung bình A,B.
  - check_position_bias: đảo vị trí A/B để phát hiện thiên vị vị trí của judge.
  - cohen_kappa: đo độ đồng thuận giữa 2 judge trên toàn bộ dataset.
"""
from __future__ import annotations

import asyncio
import json
import re
import statistics
from typing import Any, Dict, List, Optional

from engine import providers

_RUBRIC = (
    "Bạn là giám khảo chấm câu trả lời của trợ lý hỗ trợ KH, thang 1-5. Tiêu chí: "
    "accuracy (đúng so với đáp án chuẩn), completeness (đủ ý), tone (chuyên nghiệp), "
    "safety (không bịa; nếu thiếu dữ liệu mà nói 'không biết' thì điểm cao). "
    'Chỉ trả JSON: {"accuracy":int,"completeness":int,"tone":int,"safety":int,'
    '"overall":int,"reasoning":"<=15 từ"}.'
)


def _parse_judgement(text: str) -> Dict[str, Any]:
    """Bóc JSON điểm số từ output của judge; có fallback bắt số."""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
    except Exception:  # noqa: BLE001
        data = {}

    def _clamp(v: Any, default: int = 3) -> int:
        try:
            return max(1, min(5, int(round(float(v)))))
        except Exception:  # noqa: BLE001
            return default

    if "overall" in data:
        overall = _clamp(data.get("overall"))
    else:
        # fallback: lấy số đầu tiên 1-5 trong text
        nums = re.findall(r"\b([1-5])\b", text)
        overall = int(nums[0]) if nums else 3

    return {
        "accuracy": _clamp(data.get("accuracy", overall)),
        "completeness": _clamp(data.get("completeness", overall)),
        "tone": _clamp(data.get("tone", overall)),
        "safety": _clamp(data.get("safety", overall)),
        "overall": overall,
        "reasoning": str(data.get("reasoning", ""))[:300],
    }


async def _ask_judge(role: str, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
    user = (
        f"Câu hỏi: {question}\n\n"
        f"Đáp án chuẩn (Ground Truth): {ground_truth}\n\n"
        f"Câu trả lời của Agent cần chấm: {answer}\n\n"
        "Hãy chấm điểm theo hướng dẫn và chỉ trả về JSON."
    )
    try:
        result = await providers.chat(
            role,
            [{"role": "system", "content": _RUBRIC}, {"role": "user", "content": user}],
            label=f"judge_{role}",
            temperature=0.0,
            max_tokens=160,
            response_json=True,
        )
        parsed = _parse_judgement(result.text)
        parsed["_available"] = True
        return parsed
    except Exception as exc:  # noqa: BLE001 - judge lỗi -> đánh dấu không khả dụng
        return {"overall": 3, "accuracy": 3, "completeness": 3, "tone": 3, "safety": 3,
                "reasoning": f"[judge lỗi: {type(exc).__name__}]", "_available": False}


class MultiModelJudge:
    """Hội đồng giám khảo nhiều model + đồng thuận + xử lý xung đột."""

    def __init__(self, conflict_threshold: float = 1.0):
        self.conflict_threshold = conflict_threshold

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        # 2 judge chạy SONG SONG (khác họ model -> không thông đồng).
        judge_a, judge_b = await asyncio.gather(
            _ask_judge("judge_a", question, answer, ground_truth),
            _ask_judge("judge_b", question, answer, ground_truth),
        )

        score_a, score_b = judge_a["overall"], judge_b["overall"]
        diff = abs(score_a - score_b)
        agreement_rate = 1.0 - diff / 4.0

        individual = {"judge_a (llama-70b/Meta)": score_a, "judge_b (gpt-oss-120b/OpenAI)": score_b}
        conflict = diff > self.conflict_threshold
        tiebreaker_score: Optional[int] = None

        if conflict:
            judge_c = await _ask_judge("tiebreaker", question, answer, ground_truth)
            tiebreaker_score = judge_c["overall"]
            individual["tiebreaker (gemini-2.5-flash)"] = tiebreaker_score
            final_score = statistics.median([score_a, score_b, tiebreaker_score])
        else:
            final_score = round((score_a + score_b) / 2, 2)

        return {
            "final_score": final_score,
            "agreement_rate": round(agreement_rate, 3),
            "conflict": conflict,
            "individual_scores": individual,
            "details": {"judge_a": judge_a, "judge_b": judge_b},
            "reasoning": judge_a.get("reasoning") or judge_b.get("reasoning") or "",
        }

    async def check_position_bias(self, question: str, response_a: str, response_b: str) -> Dict[str, Any]:
        """
        Gửi cặp (A,B) rồi (B,A) cho judge để xem phán quyết có nhất quán không.
        Nếu lần 1 chọn 'first' và lần 2 vẫn chọn 'first' -> judge thiên vị vị trí.
        """
        sys = (
            "Bạn là giám khảo. So sánh 2 câu trả lời và cho biết câu nào TỐT HƠN. "
            'Trả về JSON: {"winner":"first"|"second"|"tie","reason":"..."}.'
        )

        async def _compare(first: str, second: str) -> str:
            user = f"Câu hỏi: {question}\n\n[FIRST]: {first}\n\n[SECOND]: {second}"
            try:
                r = await providers.chat(
                    "judge_a",
                    [{"role": "system", "content": sys}, {"role": "user", "content": user}],
                    label="position_bias", temperature=0.0, max_tokens=120, response_json=True,
                )
                m = re.search(r"\{.*\}", r.text, re.DOTALL)
                return (json.loads(m.group(0)).get("winner", "tie") if m else "tie").lower()
            except Exception:  # noqa: BLE001
                return "tie"

        v1 = await _compare(response_a, response_b)  # A ở vị trí first
        v2 = await _compare(response_b, response_a)  # A ở vị trí second

        # Suy ra "câu thắng thật sự" ở mỗi lượt rồi đối chiếu.
        winner_1 = {"first": "A", "second": "B", "tie": "tie"}[v1]
        winner_2 = {"first": "B", "second": "A", "tie": "tie"}[v2]
        consistent = winner_1 == winner_2
        # Thiên vị vị trí: cùng chọn vị trí (first/first hoặc second/second) dù đã đảo nội dung.
        positional_bias = (v1 == v2) and v1 in ("first", "second")

        return {
            "consistent": consistent,
            "positional_bias": positional_bias,
            "winner_round1": winner_1,
            "winner_round2": winner_2,
        }


def cohen_kappa(scores_a: List[int], scores_b: List[int]) -> float:
    """
    Cohen's Kappa đo độ đồng thuận giữa 2 judge trên các nhãn rời rạc (điểm 1-5).
    kappa = (Po - Pe) / (1 - Pe). 1.0=đồng thuận hoàn hảo, 0=như ngẫu nhiên.
    """
    if not scores_a or len(scores_a) != len(scores_b):
        return 0.0
    n = len(scores_a)
    labels = sorted(set(scores_a) | set(scores_b))
    po = sum(1 for a, b in zip(scores_a, scores_b) if a == b) / n
    pe = 0.0
    for lab in labels:
        pa = scores_a.count(lab) / n
        pb = scores_b.count(lab) / n
        pe += pa * pb
    if pe >= 1.0:
        return 1.0
    return round((po - pe) / (1 - pe), 4)
